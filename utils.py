import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

class EMA:
    """
    Exponential Moving Average for model parameters.
    This helps stabilize training and improves generation quality.
    """
    def __init__(self, beta=0.999):
        self.beta = beta
        self.step = 0
    
    def update_model_average(self, ma_model, current_model):
        """Update the moving average model."""
        self.step += 1
        
        # Bias correction
        decay = min(self.beta, (1 + self.step) / (10 + self.step))
        
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            if current_params.requires_grad:
                old_weight = ma_params.data
                new_weight = current_params.data
                ma_params.data = decay * old_weight + (1 - decay) * new_weight

def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2

def save_debug_image(tensors, filepath, nrow=8):
    """Save a grid of images for debugging."""
    tensors = denormalize(tensors)
    save_image(tensors, filepath, nrow=nrow)

def plot_loss_curves(loss_history, save_path):
    """Plot training loss curves."""
    if not loss_history:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Group losses by type
    d_losses = {k: v for k, v in loss_history.items() if k.startswith('D/')}
    g_losses = {k: v for k, v in loss_history.items() if k.startswith('G/')}
    
    # Plot discriminator losses
    if d_losses:
        ax = axes[0]
        for key, values in d_losses.items():
            if len(values) > 0:
                ax.plot(values, label=key)
        ax.set_title('Discriminator Losses')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    # Plot generator losses
    if g_losses:
        for idx, (key, values) in enumerate(g_losses.items(), 1):
            if idx < len(axes) and len(values) > 0:
                ax = axes[idx]
                ax.plot(values)
                ax.set_title(key)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_translation_grid(generator, style_encoder, source_images, 
                             ref_images_dict, device):
    """
    Generate a grid showing translations to multiple domains.
    
    Args:
        generator: Generator model
        style_encoder: Style encoder model
        source_images: Tensor of source images [N, 3, H, W]
        ref_images_dict: Dict mapping domain_idx to reference images
        device: Device to run on
    
    Returns:
        Grid tensor
    """
    generator.eval()
    style_encoder.eval()
    
    rows = []
    
    with torch.no_grad():
        for src_img in source_images:
            src_img = src_img.unsqueeze(0).to(device)
            row = [src_img]
            
            for domain_idx, ref_imgs in sorted(ref_images_dict.items()):
                if len(ref_imgs) > 0:
                    ref_img = ref_imgs[0].unsqueeze(0).to(device)
                    y = torch.tensor([domain_idx], device=device)
                    
                    # Extract style and generate
                    style = style_encoder(ref_img, y)
                    fake = generator(src_img, style)
                    row.append(fake)
            
            rows.append(torch.cat(row, dim=0))
    
    return torch.cat(rows, dim=0)

def compute_fid_score(real_features, fake_features):
    """
    Compute Fréchet Inception Distance between real and fake features.
    This is a placeholder - you would need to extract features using InceptionV3.
    """
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_fake)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def setup_logger(log_dir):
    """Setup logging for training."""
    import logging
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('multidomain')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger