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

class DynamicWeightScheduler:
    """
    Dynamically adjusts the weights of different loss components during training.
    This scheduler implements a warmup phase followed by a cosine decay phase,
    which can help stabilize training in the early stages and refine the model later.
    """
    def __init__(self, init_weights, warmup_epochs=10, decay_epochs=100, total_epochs=200):
        self.init_weights = init_weights
        self.current_weights = init_weights.copy()
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = warmup_epochs + decay_epochs
        self.total_epochs = total_epochs
        
        self.loss_history = {k: [] for k in init_weights.keys()}
        self.weight_history = {k: [] for k in init_weights.keys()}

    def get_current_weights(self, epoch, current_losses):
        # 1. Update loss history
        for k, v in current_losses.items():
            if k in self.loss_history:
                self.loss_history[k].append(v.detach().cpu().item())

        # 2. Calculate warmup factor
        warmup_factor = min(1.0, (epoch + 1) / self.warmup_epochs)

        # 3. Calculate decay factor (Cosine decay)
        decay_factor = 1.0
        if epoch >= self.warmup_epochs:
            progress = min(1.0, (epoch - self.warmup_epochs) / (self.decay_end_epoch - self.warmup_epochs))
            # Cosine decay from 1 down to 0
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            # Rescale to decay from 1 down to a minimum of 0.1
            decay_factor = 0.1 + 0.9 * cosine_decay

        # 4. Update weights and record history
        for k in self.current_weights.keys():
            self.current_weights[k] = self.init_weights[k] * warmup_factor * decay_factor
            self.weight_history[k].append(self.current_weights[k])
            
        return self.current_weights

    def plot_weight_history(self, save_path=None):
        """
        Plots the evolution of the loss weights over epochs and saves the plot.
        """
        if not any(self.weight_history.values()):
            return
        plt.figure(figsize=(15, 8))
        for k, v in self.weight_history.items():
            if v:
                plt.plot(v, label=k, linewidth=2)
        plt.title('Loss Weight Evolution Over Training')
        plt.xlabel('Epochs')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()