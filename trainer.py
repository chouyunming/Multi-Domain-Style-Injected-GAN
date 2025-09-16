# trainer.py
import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# W&B for experiment tracking
import wandb

# Import from our new modules
from model import StyleCycleGANGenerator, MultiDomainStyleEncoder, MultiDomainDiscriminator
from losses import VGGStyleContentLoss
from utils import EMA, DynamicWeightScheduler, save_sample_grid

class MultiDomainStyleCycleGAN:
    """
    Multi-domain StyleCycleGAN trainer.
    This extends the original StyleCycleGAN to work with multiple target domains
    using domain-specific branches in the style encoder and discriminator.
    """
    def __init__(self, device, total_epochs, lr_g, lr_d, loss_weights, num_domains):
        self.device = device
        self.num_domains = num_domains

        # --- Instantiate Models ---
        # Generators (unchanged - they work with style codes)
        self.G_A2B = StyleCycleGANGenerator().to(device)
        self.G_B2A = StyleCycleGANGenerator().to(device)
        
        # Multi-domain Style Encoders
        self.SE_A = MultiDomainStyleEncoder(num_domains=num_domains).to(device)
        self.SE_B = MultiDomainStyleEncoder(num_domains=num_domains).to(device)
        
        # Multi-domain Discriminators
        self.D_A = MultiDomainDiscriminator(num_domains=num_domains).to(device)
        self.D_B = MultiDomainDiscriminator(num_domains=num_domains).to(device)

        # --- Instantiate EMA Models for Inference ---
        self.ema = EMA(beta=0.995)
        self.ema_G_A2B = copy.deepcopy(self.G_A2B).eval()
        self.ema_G_B2A = copy.deepcopy(self.G_B2A).eval()
        self.ema_SE_A = copy.deepcopy(self.SE_A).eval()
        self.ema_SE_B = copy.deepcopy(self.SE_B).eval()

        # --- Define Loss Functions ---
        self.criterion_gan = nn.MSELoss().to(device)
        self.criterion_cycle = nn.L1Loss().to(device)
        self.criterion_identity = nn.L1Loss().to(device)
        self.criterion_style_content = VGGStyleContentLoss(device).to(device)

        # --- Define Optimizers ---
        g_params = list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + \
                   list(self.SE_A.parameters()) + list(self.SE_B.parameters())
        self.g_optimizer = torch.optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999))
        
        d_params = list(self.D_A.parameters()) + list(self.D_B.parameters())
        self.d_optimizer = torch.optim.Adam(d_params, lr=lr_d, betas=(0.5, 0.999))

        # --- Define Learning Rate Schedulers ---
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=total_epochs, eta_min=1e-6)
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_optimizer, T_max=total_epochs, eta_min=1e-6)
        
        # --- Initialize Dynamic Loss Weighting ---
        self.weight_scheduler = DynamicWeightScheduler(loss_weights, warmup_epochs=10, decay_epochs=100, total_epochs=total_epochs)

        # --- Loss History Tracking ---
        self.loss_history = {k: [] for k in (list(loss_weights.keys()) + ['D_loss', 'G_loss'])}
        self.current_epoch_losses = {k: [] for k in self.loss_history.keys()}

    def train_step(self, batch, epoch):
        """
        Training step modified to handle multi-domain data.
        """
        # Extract data from batch
        real_A = batch['source'].to(self.device)  # Source domain images
        real_B = batch['target'].to(self.device)  # Target domain images
        y_org = batch['source_domain'].to(self.device)  # Source domain indices (always 0)
        y_trg = batch['target_domain'].to(self.device)  # Target domain indices
        
        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), 1, *self.D_A(real_A, y_org).shape[2:]), device=self.device)
        fake = torch.zeros_like(valid)

        # ==================================
        #        Train Generators
        # ==================================
        self.g_optimizer.zero_grad()
        
        # Extract style codes using domain indices
        style_A = self.SE_A(real_A, y_org)  # Style of source images
        style_B = self.SE_B(real_B, y_trg)  # Style of target images

        # Identity loss: G(B, style_B) should be B for same domain
        # Only compute identity loss for target domain images
        loss_identity = self.criterion_identity(self.G_A2B(real_B, style_B), real_B)

        # GAN loss, Style loss, and Content loss
        fake_B = self.G_A2B(real_A, style_B)  # Translate A to B's style
        loss_gan_A2B = self.criterion_gan(self.D_B(fake_B, y_trg), valid)
        content_loss_B, style_loss_B = self.criterion_style_content(fake_B, real_B, real_A)
        
        # For reverse direction, we translate back to source domain
        fake_A = self.G_B2A(real_B, style_A)  # Translate B to A's style
        loss_gan_B2A = self.criterion_gan(self.D_A(fake_A, y_org), valid)
        content_loss_A, style_loss_A = self.criterion_style_content(fake_A, real_A, real_B)
        
        loss_gan = (loss_gan_A2B + loss_gan_B2A) / 2
        loss_style = (style_loss_A + style_loss_B) / 2
        loss_content = (content_loss_A + content_loss_B) / 2

        # Cycle-consistency loss
        loss_cycle = (self.criterion_cycle(self.G_B2A(fake_B, style_A), real_A) + \
                      self.criterion_cycle(self.G_A2B(fake_A, style_B), real_B)) / 2
        
        # Total generator loss with dynamic weights
        individual_losses = {'gan': loss_gan, 'cycle': loss_cycle, 'identity': loss_identity, 
                           'style': loss_style, 'content': loss_content}
        weights = self.weight_scheduler.get_current_weights(epoch, individual_losses)
        g_loss = sum(loss * weights[name] for name, loss in individual_losses.items())
        
        g_loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.g_optimizer.param_groups[0]['params'], 1.0)
        self.g_optimizer.step()

        # Update EMA models after generator step
        self.ema.update_model_average(self.ema_G_A2B, self.G_A2B)
        self.ema.update_model_average(self.ema_G_B2A, self.G_B2A)
        self.ema.update_model_average(self.ema_SE_A, self.SE_A)
        self.ema.update_model_average(self.ema_SE_B, self.SE_B)

        # ==================================
        #      Train Discriminators
        # ==================================
        self.d_optimizer.zero_grad()
        
        # Real loss
        loss_real_A = self.criterion_gan(self.D_A(real_A, y_org), valid)
        loss_real_B = self.criterion_gan(self.D_B(real_B, y_trg), valid)
        
        # Fake loss
        loss_fake_A = self.criterion_gan(self.D_A(fake_A.detach(), y_org), fake)
        loss_fake_B = self.criterion_gan(self.D_B(fake_B.detach(), y_trg), fake)
        
        d_loss = (loss_real_A + loss_fake_A + loss_real_B + loss_fake_B) / 2
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.d_optimizer.param_groups[0]['params'], 1.0)
        self.d_optimizer.step()

        return {'D_loss': d_loss, 'G_loss': g_loss, **individual_losses}

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Save main models and optimizers
        torch.save({
            'G_A2B': self.G_A2B.state_dict(), 'G_B2A': self.G_B2A.state_dict(),
            'SE_A': self.SE_A.state_dict(), 'SE_B': self.SE_B.state_dict(),
            'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(), 'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(), 'd_scheduler': self.d_scheduler.state_dict(),
            'loss_history': self.loss_history,
            'num_domains': self.num_domains
        }, os.path.join(save_dir, 'checkpoint.pth'))
        # Save EMA models separately
        torch.save({
            'ema_G_A2B': self.ema_G_A2B.state_dict(), 'ema_G_B2A': self.ema_G_B2A.state_dict(),
            'ema_SE_A': self.ema_SE_A.state_dict(), 'ema_SE_B': self.ema_SE_B.state_dict()
        }, os.path.join(save_dir, 'ema_checkpoint.pth'))
        print(f"Models successfully saved to {save_dir}")

    def load_models(self, checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return 0
            
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Verify domain compatibility
        saved_num_domains = ckpt.get('num_domains', 2)  # Default for backward compatibility
        if saved_num_domains != self.num_domains:
            print(f"Warning: Saved model has {saved_num_domains} domains, but current model expects {self.num_domains}")
            return 0
            
        self.G_A2B.load_state_dict(ckpt['G_A2B']); self.G_B2A.load_state_dict(ckpt['G_B2A'])
        self.SE_A.load_state_dict(ckpt['SE_A']); self.SE_B.load_state_dict(ckpt['SE_B'])
        self.D_A.load_state_dict(ckpt['D_A']); self.D_B.load_state_dict(ckpt['D_B'])
        self.g_optimizer.load_state_dict(ckpt['g_optimizer']); self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        self.g_scheduler.load_state_dict(ckpt['g_scheduler']); self.d_scheduler.load_state_dict(ckpt['d_scheduler'])
        self.loss_history = ckpt.get('loss_history', self.loss_history)

        # Load EMA models
        ema_ckpt_path = os.path.join(checkpoint_dir, 'ema_checkpoint.pth')
        if os.path.exists(ema_ckpt_path):
            ema_ckpt = torch.load(ema_ckpt_path, map_location=self.device)
            self.ema_G_A2B.load_state_dict(ema_ckpt['ema_G_A2B']); self.ema_G_B2A.load_state_dict(ema_ckpt['ema_G_B2A'])
            self.ema_SE_A.load_state_dict(ema_ckpt['ema_SE_A']); self.ema_SE_B.load_state_dict(ema_ckpt['ema_SE_B'])
        
        print(f"Models successfully loaded from {checkpoint_dir}")
        start_epoch = len(self.loss_history.get('G_loss', []))
        return start_epoch
        
    def plot_losses(self, save_path):
        if not self.loss_history or not any(v for k, v in self.loss_history.items() if k in ['G_loss', 'D_loss']): return
        plt.figure(figsize=(12, 8))
        epochs = range(1, len(self.loss_history['G_loss']) + 1)
        for loss_type, values in self.loss_history.items():
            if values: plt.plot(epochs, values, label=loss_type)
        plt.legend(); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Losses Over Epochs')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(save_path, dpi=300); plt.close()

    def generate_cyclegan_style_grid(self, batch):
        """Generate single 2x2 grid: Real A, Fake B, Real B, Fake A"""
        with torch.no_grad():
            # Get first sample from batch
            real_A = batch['source'][0:1].to(self.device)  # Source
            real_B = batch['target'][0:1].to(self.device)  # Target
            y_org = batch['source_domain'][0:1].to(self.device)  # Source domain 
            y_trg = batch['target_domain'][0:1].to(self.device)  # Target domain
            
            # Extract styles
            style_A = self.ema_SE_A(real_A, y_org)  # Style of source
            style_B = self.ema_SE_B(real_B, y_trg)  # Style of target
            
            # Generate translations
            fake_B = self.ema_G_A2B(real_A, style_B)  # A → B
            fake_A = self.ema_G_B2A(real_B, style_A)  # B → A
            
            # Create 2x2 grid: [Real A, Fake B, Real B, Fake A]
            grid = torch.cat([real_A, fake_B, real_B, fake_A], dim=0)
            
            return grid, y_trg[0].item()
        """Generate Real/Fake pairs for a specific target domain (CycleGAN style)"""
        with torch.no_grad():
            # Get source images (limited number for cleaner visualization)
            source_images = torch.stack(fixed_samples['source'][:num_samples]).to(self.device)
            
            # Get real target domain images
            if target_domain_idx in fixed_samples['targets'] and len(fixed_samples['targets'][target_domain_idx]) > 0:
                real_target_images = torch.stack(
                    fixed_samples['targets'][target_domain_idx][:num_samples]
                ).to(self.device)
                
                # Use first target image as style reference
                ref_img = real_target_images[:1]
                domain_tensor = torch.tensor([target_domain_idx], device=self.device)
                style_B = self.ema_SE_B(ref_img, domain_tensor)
                
                # Generate fake images by translating source to target domain style
                fake_images = []
                for src_img in source_images:
                    fake = self.ema_G_A2B(src_img.unsqueeze(0), style_B)
                    fake_images.append(fake)
                fake_batch = torch.cat(fake_images, dim=0)
                
                # Create Real/Fake pairs: [Real1, Fake1, Real2, Fake2, ...]
                samples = []
                for i in range(num_samples):
                    if i < len(real_target_images):
                        samples.append(real_target_images[i:i+1])
                    if i < len(fake_batch):
                        samples.append(fake_batch[i:i+1])
                
                return torch.cat(samples, dim=0) if samples else source_images
            
            return source_images


def train_multi_domain_style_cyclegan(model, dataset, cfg, start_epoch=0):
    """
    The main training loop for multi-domain StyleCycleGAN.
    """
    save_dir = os.path.join(cfg.save_dir_base, cfg.EXPERIMENT_NAME)
    images_dir = os.path.join(save_dir, 'images')
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # W&B: Watch models to log gradients and parameters
    if cfg.wandb:
        wandb.watch(models=(model.G_A2B, model.G_B2A, model.SE_A, model.SE_B, model.D_A, model.D_B), log_freq=50)
    
    for epoch in range(start_epoch, cfg.epochs):
        # Reset per-epoch loss storage
        for k in model.current_epoch_losses.keys(): model.current_epoch_losses[k] = []
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{cfg.epochs}')
        
        for i, batch in pbar:
            losses = model.train_step(batch, epoch)
            # Store batch losses for epoch average
            for k, v in losses.items(): model.current_epoch_losses[k].append(v.item())
            pbar.set_postfix({k: f'{v.item():.3f}' for k, v in losses.items()})
            
            # W&B: Log step-wise losses
            if cfg.wandb:
                step_logs = {f"loss/{k}": v.item() for k, v in losses.items()}
                wandb.log(step_logs)

            # Save sample images periodically - Classic CycleGAN 2x2 format
            if i % cfg.save_freq == 0:
                # Generate single 2x2 grid from current batch
                grid, target_domain_idx = model.generate_cyclegan_style_grid(batch)
                domain_name = dataset.domains[target_domain_idx] if target_domain_idx < len(dataset.domains) else f"Domain_{target_domain_idx}"
                
                # Labels for 2x2 grid: Real A, Fake B, Real B, Fake A
                labels = [
                    f"Real A ({dataset.domains[0]})",     # Top-left: Source
                    f"Fake B ({domain_name})",            # Top-right: A→B  
                    f"Real B ({domain_name})",            # Bottom-left: Target
                    f"Fake A ({dataset.domains[0]})"     # Bottom-right: B→A
                ]
                
                save_sample_grid(
                    grid,
                    os.path.join(images_dir, f'epoch_{epoch+1:03d}_batch_{i:04d}_{domain_name}.png'),
                    nrow=2,  # 2x2 grid
                    domain_names=labels
                )

        # Calculate and store average losses for the epoch
        epoch_avg_losses = {f"avg_loss/{k}": np.mean(v_list) for k, v_list in model.current_epoch_losses.items() if v_list}
        for k, v_list in model.current_epoch_losses.items():
            if v_list: model.loss_history[k].append(np.mean(v_list))

        # W&B: Log epoch-level metrics
        if cfg.wandb:
            epoch_logs = { "epoch": epoch + 1, **epoch_avg_losses }
            epoch_logs["lr/generator"] = model.g_scheduler.get_last_lr()[0]
            epoch_logs["lr/discriminator"] = model.d_scheduler.get_last_lr()[0]
            for name, weight in model.weight_scheduler.current_weights.items():
                epoch_logs[f"weight/{name}"] = weight
            wandb.log(epoch_logs)

        # Step the LR schedulers
        model.g_scheduler.step(); model.d_scheduler.step()
        
        # Generate and save loss plots
        model.plot_losses(os.path.join(save_dir, 'losses.png'))
        model.weight_scheduler.plot_weight_history(os.path.join(save_dir, 'weight_history.png'))

        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
            checkpoint_dir = os.path.join(checkpoints_dir, f'epoch_{epoch+1}')
            model.save_models(checkpoint_dir)

    print("Multi-domain training completed!")