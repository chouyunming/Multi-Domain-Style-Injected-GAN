import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import wandb

from model import build_model
from losses import VGGStyleContentLoss
from utils import EMA


class MSIGAN:
    """
    Multi-domain trainer class - FIXED version with proper domain handling
    """
    def __init__(self, img_size, style_dim, num_domains, device, lr_g, lr_d, 
                 loss_weights, total_epochs):
        self.device = device
        self.num_domains = num_domains
        self.loss_weights = loss_weights
        
        # Build models
        self.generator, self.style_encoder, self.discriminator = build_model(
            img_size, style_dim, num_domains, device
        )
        
        # EMA models for stable inference
        self.ema = EMA(beta=0.999)
        self.ema_generator = copy.deepcopy(self.generator).eval()
        self.ema_style_encoder = copy.deepcopy(self.style_encoder).eval()
        
        # Loss functions
        self.criterion_vgg = VGGStyleContentLoss(device)
        
        # Optimizers
        g_params = list(self.generator.parameters()) + list(self.style_encoder.parameters())
        self.g_optimizer = torch.optim.Adam(g_params, lr=lr_g, betas=(0.0, 0.99))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99)
        )
        
        # Learning rate schedulers
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=total_epochs, eta_min=1e-6
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=total_epochs, eta_min=1e-6
        )
        
        # Loss tracking
        self.loss_history = {}
        self.current_epoch = 0
    
    def compute_d_loss(self, x_real, y_org, x_ref, y_trg):
        """Compute discriminator loss with proper shape handling."""
        # Real images from source domain
        x_real.requires_grad_()
        out_real = self.discriminator(x_real, y_org)
        
        # Ensure output has correct shape for loss computation
        if out_real.dim() == 1:
            out_real = out_real.unsqueeze(-1)  # [B] -> [B, 1]
        
        # Use BCEWithLogitsLoss for stability
        real_labels = torch.ones_like(out_real) #* 0.9  # Label smoothing
        loss_real = F.binary_cross_entropy_with_logits(out_real, real_labels)
        
        # Fake images translated to target domain
        with torch.no_grad():
            s_trg = self.style_encoder(x_ref, y_trg)
            x_fake = self.generator(x_real, s_trg)
        
        out_fake = self.discriminator(x_fake, y_trg)
        if out_fake.dim() == 1:
            out_fake = out_fake.unsqueeze(-1)
        
        fake_labels = torch.zeros_like(out_fake) #+ 0.1  # Label smoothing
        loss_fake = F.binary_cross_entropy_with_logits(out_fake, fake_labels)
        
        # Gradient penalty for real images
        loss_gp = self.gradient_penalty(out_real, x_real)
        
        loss = loss_real + loss_fake + 1.0 * loss_gp
        
        return loss, {
            'D/real': loss_real.item(),
            'D/fake': loss_fake.item(),
            'D/gp': loss_gp.item()
        }

    def compute_g_loss(self, x_real, y_org, x_ref, y_trg, x_ref2):
        """Compute generator and style encoder losses."""
        losses = {}
        
        # Forward translation: source -> target
        s_trg = self.style_encoder(x_ref, y_trg)
        x_fake = self.generator(x_real, s_trg)
        
        # Style reconstruction loss
        if self.loss_weights.get('style_recon', 0) > 0 or self.loss_weights.get('style_div', 0) > 0:
            s_pred = self.style_encoder(x_fake, y_trg)
            loss_sty = torch.mean(torch.abs(s_pred - s_trg))
            losses['style_recon'] = loss_sty
        
        # Style diversification loss
            s_trg2 = self.style_encoder(x_ref2, y_trg)
            x_fake2 = self.generator(x_real, s_trg2)
            x_fake2 = x_fake2.detach()
            loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
            losses['style_div'] = -loss_ds  # Negative because we want to maximize diversity
        
        else:
            losses['style_recon'] = torch.tensor(0.0, device=self.device)
            losses['style_div'] = torch.tensor(0.0, device=self.device)

        # Cycle consistency loss (reconstruct source)
        s_org = self.style_encoder(x_real, y_org)
        x_recon = self.generator(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_recon - x_real))
        losses['cycle'] = loss_cyc
        
        # Identity loss (preserve content when style is from same domain)
        x_idt = self.generator(x_real, s_org)  # Apply source style to source image
        loss_idt = torch.mean(torch.abs(x_idt - x_real))
        losses['identity'] = loss_idt
        
        # Adversarial loss
        out_fake = self.discriminator(x_fake, y_trg)
        if out_fake.dim() == 1:
            out_fake = out_fake.unsqueeze(-1)
        real_labels = torch.ones_like(out_fake)
        loss_adv = F.binary_cross_entropy_with_logits(out_fake, real_labels)
        losses['gan'] = loss_adv
        
        # VGG perceptual losses - only compute if weights > 0
        if self.loss_weights.get('content', 0) > 0 or self.loss_weights.get('style', 0) > 0:
            content_loss, style_loss = self.criterion_vgg(x_fake, x_ref, x_real)
            losses['content'] = content_loss
            losses['style'] = style_loss
        else:
            losses['content'] = torch.tensor(0.0, device=self.device)
            losses['style'] = torch.tensor(0.0, device=self.device)
        
        # Total generator loss
        total_loss = sum(
            self.loss_weights.get(name, 0.0) * loss 
            for name, loss in losses.items()
            if self.loss_weights.get(name, 0.0) != 0  # Skip zero-weighted losses
        )
        
        return total_loss, losses

    def gradient_penalty(self, d_out, x_in):
        """Compute gradient penalty for discriminator with correct shape handling."""
        batch_size = x_in.size(0)
        
        # Handle different output shapes
        if d_out.dim() > 1:
            d_out = d_out.mean()  # Reduce to scalar for gradient computation
        
        grad_dout = torch.autograd.grad(
            outputs=d_out,
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
    
    def train_step(self, batch):
        """Single training step - FIXED to use all batch data."""
        x_real = batch['source'].to(self.device)
        x_ref = batch['target'].to(self.device)
        x_ref2 = batch['target2'].to(self.device)  # Use second reference
        y_org = batch['source_domain'].to(self.device)  # Source domain index (always 0)
        y_trg = batch['target_domain'].to(self.device)  # Target domain index
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        d_loss, d_losses = self.compute_d_loss(x_real, y_org, x_ref, y_trg)
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train generator (with both references for style diversification)
        self.g_optimizer.zero_grad()
        g_loss, g_losses = self.compute_g_loss(x_real, y_org, x_ref, y_trg, x_ref2)
        g_loss.backward()
        self.g_optimizer.step()
        
        # Update EMA
        self.ema.update_model_average(self.ema_generator, self.generator)
        self.ema.update_model_average(self.ema_style_encoder, self.style_encoder)
        
        # Combine losses for logging
        all_losses = {**d_losses}
        for name, loss in g_losses.items():
            all_losses[f'G/{name}'] = loss.item() if torch.is_tensor(loss) else loss
        all_losses['G/total'] = g_loss.item()
        
        return all_losses
    
    def save_checkpoint(self, save_dir, epoch):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ema_generator': self.ema_generator.state_dict(),
            'ema_style_encoder': self.ema_style_encoder.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(),
            'loss_history': self.loss_history,
        }
        
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.ema_generator.load_state_dict(checkpoint['ema_generator'])
        self.ema_style_encoder.load_state_dict(checkpoint['ema_style_encoder'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler'])
        self.loss_history = checkpoint.get('loss_history', {})
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return start_epoch


def train(model, dataset, cfg, start_epoch=0):
    """Main training loop - FIXED with proper sample generation."""
    # Setup directories
    save_dir = os.path.join(cfg.save_dir_base, cfg.EXPERIMENT_NAME)
    sample_dir = os.path.join(save_dir, 'samples')
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    # Get fixed samples for visualization
    fixed_samples = dataset.get_fixed_samples(num_samples=4)
    
    # W&B: Watch models
    if cfg.wandb:
        wandb.watch(
            models=(model.generator, model.style_encoder, model.discriminator),
            log_freq=100
        )
    
    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        epoch_losses = {}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
        
        for i, batch in enumerate(pbar):
            # Training step
            losses = model.train_step(batch)
            
            # Update progress bar
            pbar.set_postfix({k: f'{v:.3f}' for k, v in losses.items()})
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # W&B: Log step losses
            if cfg.wandb and i % 10 == 0:
                wandb.log({f'step/{k}': v for k, v in losses.items()})
            
            # Generate samples
            if i % cfg.save_freq == 0:
                with torch.no_grad():
                    # Use current batch data for more diverse samples
                    batch_data = next(iter(dataloader))
                    
                    all_rows = []
                    num_samples_per_domain = 2
                    
                    # For each source sample
                    for src_idx in range(min(num_samples_per_domain, batch_data['source'].size(0))):
                        source = batch_data['source'][src_idx:src_idx+1].to(model.device)
                        
                        # Create row: source + translations to each target domain
                        row = [source]
                        
                        # Generate translation to each target domain
                        for domain_idx in range(1, dataset.num_domains):  # Skip source domain (0)
                            # Get a random reference from this domain
                            if domain_idx in fixed_samples['targets'] and len(fixed_samples['targets'][domain_idx]) > 0:
                                # Use different reference images for variety
                                ref_idx = i % len(fixed_samples['targets'][domain_idx])
                                ref = fixed_samples['targets'][domain_idx][ref_idx].unsqueeze(0).to(model.device)
                                y_trg = torch.tensor([domain_idx], device=model.device)
                                
                                # Extract style and generate
                                s = model.ema_style_encoder(ref, y_trg)
                                fake = model.ema_generator(source, s)
                                row.append(fake)
                        
                        if len(row) > 1:  # Only add if we have translations
                            all_rows.append(torch.cat(row, dim=0))
                    
                    if all_rows:
                        samples_grid = torch.cat(all_rows, dim=0)
                        
                        # Save with informative filename
                        save_image(
                            samples_grid,
                            os.path.join(sample_dir, f'epoch_{epoch+1:03d}_iter_{i:05d}.png'),
                            nrow=dataset.num_domains,  # One column per domain
                            normalize=True,
                            value_range=(-1, 1)
                        )
                        
                        # Also save individual translations for better inspection
                        if epoch % 10 == 0:  # Every 10 epochs
                            for row_idx, row_tensor in enumerate(all_rows):
                                save_image(
                                    row_tensor,
                                    os.path.join(sample_dir, f'epoch_{epoch+1:03d}_sample_{row_idx}.png'),
                                    nrow=dataset.num_domains,
                                    normalize=True,
                                    value_range=(-1, 1)
                                )
                    
                        # # W&B: Log samples
                        # if cfg.wandb:
                        #     wandb.log({
                        #         'samples': wandb.Image(samples_grid, caption=f'Epoch {epoch+1}')
                        #     })
        
        # Epoch-level operations
        # Calculate average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Store in history
        for k, v in avg_losses.items():
            if k not in model.loss_history:
                model.loss_history[k] = []
            model.loss_history[k].append(v)
        
        # W&B: Log epoch metrics
        if cfg.wandb:
            epoch_log = {'epoch': epoch + 1}
            epoch_log.update({f'epoch/{k}': v for k, v in avg_losses.items()})
            epoch_log['lr/generator'] = model.g_scheduler.get_last_lr()[0]
            epoch_log['lr/discriminator'] = model.d_scheduler.get_last_lr()[0]
            wandb.log(epoch_log)
        
        # Step schedulers
        model.g_scheduler.step()
        model.d_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1:03d}')
            model.save_checkpoint(checkpoint_path, epoch + 1)
        
        # Plot loss curves
        if len(model.loss_history) > 0:
            plt.figure(figsize=(12, 8))
            for key, values in model.loss_history.items():
                if len(values) > 0:
                    plt.plot(values, label=key)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
            plt.close()
    
    print("Training completed!")