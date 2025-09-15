import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer."""
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        
    def forward(self, x, s):
        """
        x: content features [B, C, H, W]
        s: style code [B, style_dim]
        """
        h = self.fc(s)
        h = h.view(h.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlk(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, dim_in, dim_out, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        self.actv = nn.LeakyReLU(0.2)
        
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
    
    def forward(self, x):
        residual = x
        
        out = self.actv(self.norm1(x))
        out = self.conv1(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            residual = F.avg_pool2d(residual, 2)
        
        out = self.actv(self.norm2(out))
        out = self.conv2(out)
        
        if self.learned_sc:
            residual = self.conv1x1(residual)
            
        return (out + residual) / np.sqrt(2)


class AdaINResBlk(nn.Module):
    """Residual block with AdaIN for style injection."""
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(dim_in, style_dim)
        self.norm2 = AdaIN(dim_out, style_dim)
        self.actv = nn.LeakyReLU(0.2)
        
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
    
    def forward(self, x, s):
        """
        x: input features
        s: style code
        """
        residual = x
        
        out = self.actv(self.norm1(x, s))
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        out = self.conv1(out)
        out = self.actv(self.norm2(out, s))
        out = self.conv2(out)
        
        if self.learned_sc:
            residual = self.conv1x1(residual)
            
        return (out + residual) / np.sqrt(2)


class Generator(nn.Module):
    """
    Multi-domain generator that translates images using style codes.
    Architecture inspired by StarGAN v2.
    """
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        self.img_size = img_size
        self.style_dim = style_dim
        
        # From RGB
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        
        # Encoder (downsampling)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        
        # Calculate number of downsampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, downsample=True))
            self.decode.insert(0, AdaINResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out
        
        # Bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out))
            self.decode.insert(0, AdaINResBlk(dim_out, dim_out, style_dim))
        
        # To RGB
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, x, s):
        """
        x: input image [B, 3, H, W]
        s: style code [B, style_dim]
        """
        # Encode
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        
        # Decode with style injection
        for block in self.decode:
            x = block(x, s)
        
        return self.to_rgb(x)


class StyleEncoder(nn.Module):
    """
    Multi-domain style encoder with separate output branches for each domain.
    """
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        
        # Shared layers
        shared_layers = []
        shared_layers.append(nn.Conv2d(3, dim_in, 3, 1, 1))
        
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            shared_layers.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        
        shared_layers.append(nn.LeakyReLU(0.2))
        shared_layers.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        shared_layers.append(nn.LeakyReLU(0.2))
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Domain-specific output branches
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))
    
    def forward(self, x, y):
        """
        x: input image [B, 3, H, W]
        y: domain indices [B] (LongTensor)
        Returns: style codes [B, style_dim]
        """
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        
        # Collect outputs from all branches
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # [B, num_domains, style_dim]
        
        # Select the appropriate style code for each sample
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # [B, style_dim]
        
        return s


class Discriminator(nn.Module):
    """
    Multi-task discriminator for multiple domains.
    Fixed to handle feature map sizes correctly.
    """
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        
        # Shared convolutional layers
        blocks = []
        blocks.append(nn.Conv2d(3, dim_in, 3, 1, 1))
        
        # Calculate number of downsampling blocks more carefully
        # We want to end up with at least 4x4 spatial dimensions
        min_feat_size = 4
        repeat_num = int(np.log2(img_size // min_feat_size))
        
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        
        blocks.append(nn.LeakyReLU(0.2))
        
        # Instead of using a 4x4 conv that requires exactly 4x4 input,
        # use adaptive pooling to handle any size, then a 1x1 conv
        blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        blocks.append(nn.Conv2d(dim_out, dim_out, 1, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        
        self.shared = nn.Sequential(*blocks)
        
        # Domain-specific output branches
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, 1))  # Use Linear instead of Conv2d
    
    def forward(self, x, y):
        """
        x: input image [B, 3, H, W]
        y: domain indices [B] (LongTensor)
        Returns: discrimination scores [B]
        """
        h = self.shared(x)  # [B, dim_out, 1, 1]
        h = h.view(h.size(0), -1)  # [B, dim_out]
        
        # Collect outputs from all branches
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # [B, num_domains, 1]
        
        # Select the appropriate output for each sample
        idx = torch.arange(y.size(0), device=y.device)
        out = out[idx, y].squeeze(-1)  # [B]
        
        return out

def build_model(img_size, style_dim, num_domains, device):
    """Build all model components."""
    generator = Generator(img_size, style_dim).to(device)
    style_encoder = StyleEncoder(img_size, style_dim, num_domains).to(device)
    discriminator = Discriminator(img_size, num_domains).to(device)
    
    return generator, style_encoder, discriminator