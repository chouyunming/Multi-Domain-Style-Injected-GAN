import torch
import torch.nn as nn
import config as default_config

# ######################################################################
# ###########  Building Blocks for the Style-based Generator ###########
# ######################################################################

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) block.
    This layer adjusts the style of the content features based on a given style code.
    """
    def __init__(self, content_channels, style_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(content_channels, affine=False)
        # A linear layer to produce the modulation parameters (gamma and beta)
        self.style_modulation = nn.Linear(style_dim, content_channels * 2)

    def forward(self, content_features, style_code):
        normalized_content = self.instance_norm(content_features)
        
        # Handle both [B, D] and [B, D, 1, 1] style code formats
        if len(style_code.shape) == 4:
            style_code = style_code.squeeze(-1).squeeze(-1)
        
        # Split the style parameters into gamma and beta
        style_params = self.style_modulation(style_code)
        gamma, beta = style_params.chunk(2, dim=1)
        
        # Reshape gamma and beta to match content feature dimensions for broadcasting
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        
        # Apply the style modulation
        return gamma * normalized_content + beta

class ResidualBlockWithAdaIN(nn.Module):
    """
    A residual block that incorporates AdaIN layers.
    This allows the style to be injected at multiple points in the generator.
    """
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.adain1 = AdaIN(channels, style_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.adain2 = AdaIN(channels, style_dim)

    def forward(self, x, style_code):
        residual = x
        out = self.relu1(self.adain1(self.conv1(x), style_code))
        out = self.adain2(self.conv2(out), style_code)
        return out + residual

# ######################################################################
# ###################  Main Network Architectures ######################
# ######################################################################

class MultiDomainStyleEncoder(nn.Module):
    """
    Multi-domain style encoder with domain-specific branches.
    Each domain has its own output branch while sharing lower-level features.
    """
    def __init__(self, style_dim=256, num_domains=2):
        super().__init__()
        self.num_domains = num_domains
        
        # Shared convolutional layers
        self.shared_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Domain-specific output branches
        self.domain_branches = nn.ModuleList()
        for _ in range(num_domains):
            self.domain_branches.append(
                nn.Sequential(
                    nn.Conv2d(512, style_dim, kernel_size=1),
                    nn.Flatten()
                )
            )

    def forward(self, img, domain_idx=None):
        """
        Args:
            img: input image [B, 3, H, W]
            domain_idx: domain indices [B] (LongTensor) or None for single domain
        Returns:
            style codes [B, style_dim]
        """
        # Extract shared features
        shared_features = self.shared_layers(img)
        
        if domain_idx is None:
            # Single domain case - use first branch
            return self.domain_branches[0](shared_features)
        
        # Multi-domain case - select appropriate branch for each sample
        batch_size = img.size(0)
        style_codes = []
        
        # Collect outputs from all branches
        all_outputs = []
        for branch in self.domain_branches:
            all_outputs.append(branch(shared_features))
        all_outputs = torch.stack(all_outputs, dim=1)  # [B, num_domains, style_dim]
        
        # Select the appropriate output for each sample
        idx = torch.arange(batch_size, device=img.device)
        style_codes = all_outputs[idx, domain_idx]  # [B, style_dim]
        
        return style_codes


class StyleCycleGANGenerator(nn.Module):
    """
    The main generator network for StyleCycleGAN.
    It separates content and style, encoding the content and then injecting
    the style via AdaIN residual blocks in the decoder.
    """
    def __init__(self, in_channels=3, out_channels=3, style_dim=256, n_residual_blocks=default_config.N_RESIDUAL_BLOCKS):
        super().__init__()
        # Content Encoder: Extracts style-invariant content features
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True)
        )
        
        # Decoder: Synthesizes an image from content features and a style code
        decoder_blocks = [ResidualBlockWithAdaIN(256, style_dim) for _ in range(n_residual_blocks)]
        decoder_blocks.extend([
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, 1, 3, padding_mode='reflect'), nn.Tanh()
        ])
        self.decoder = nn.ModuleList(decoder_blocks)

    def forward(self, content_image, style_code):
        content_features = self.content_encoder(content_image)
        x = content_features
        for layer in self.decoder:
            # Apply style code only to the AdaIN residual blocks
            x = layer(x, style_code) if isinstance(layer, ResidualBlockWithAdaIN) else layer(x)
        return x


class MultiDomainDiscriminator(nn.Module):
    """
    Multi-domain PatchGAN-style discriminator with domain-specific branches.
    Each domain has its own output head while sharing feature extraction.
    """
    def __init__(self, in_channels=3, num_domains=2):
        super().__init__()
        self.num_domains = num_domains
        
        # Shared feature extraction layers
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize: 
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.shared_layers = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False), 
            *discriminator_block(64, 128), 
            *discriminator_block(128, 256), 
            *discriminator_block(256, 512),
        )
        
        # Domain-specific output branches
        self.domain_branches = nn.ModuleList()
        for _ in range(num_domains):
            self.domain_branches.append(nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)), 
                nn.Conv2d(512, 1, 4, padding=1)
            ))

    def forward(self, img, domain_idx=None):
        """
        Args:
            img: input image [B, 3, H, W]
            domain_idx: domain indices [B] (LongTensor) or None for single domain
        Returns:
            discrimination scores [B, 1, H, W] or [B] depending on domain_idx
        """
        # Extract shared features
        shared_features = self.shared_layers(img)
        
        if domain_idx is None:
            # Single domain case - use first branch
            return self.domain_branches[0](shared_features)
        
        # Multi-domain case - select appropriate branch for each sample
        batch_size = img.size(0)
        
        # Collect outputs from all branches
        all_outputs = []
        for branch in self.domain_branches:
            all_outputs.append(branch(shared_features))
        all_outputs = torch.stack(all_outputs, dim=1)  # [B, num_domains, 1, H, W]
        
        # Select the appropriate output for each sample
        idx = torch.arange(batch_size, device=img.device)
        outputs = all_outputs[idx, domain_idx]  # [B, 1, H, W]
        
        return outputs