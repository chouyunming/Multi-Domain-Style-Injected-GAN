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
from utils import EMA, DynamicWeightScheduler

class MSIGAN:
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
        
        # Initialize DynamicWeightScheduler
        scheduled_weights = {k: v for k, v in loss_weights.items() if k != 'gan'}
        self.weight_scheduler = DynamicWeightScheduler(
            init_weights=scheduled_weights,
            warmup_epochs=20,
            decay_epochs=100,
            total_epochs=total_epochs
        )
        
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
        
        self.loss_history = {}
        self.current_epoch = 0
    
    def compute_d_loss(self, x_real, y_org, x_ref, y_trg):
        """Computes discriminator loss"""
        x_real.requires_grad_()
        out_real = self.discriminator(x_real, y_org)
        
        if out_real.dim() == 1:
            out_real = out_real.unsqueeze(-1)
        
        real_labels = torch.ones_like(out_real) * 0.9
        loss_real = F.binary_cross_entropy_with_logits(out_real, real_labels)
        
        with torch.no_grad():
            s_trg = self.style_encoder(x_ref, y_trg)
            x_fake = self.generator(x_real, s_trg)
        
        out_fake = self.discriminator(x_fake, y_trg)
        if out_fake.dim() == 1:
            out_fake = out_fake.unsqueeze(-1)
        
        fake_labels = torch.zeros_like(out_fake) + 0.1
        loss_fake = F.binary_cross_entropy_with_logits(out_fake, fake_labels)
        
        loss_gp = self.gradient_penalty(out_real, x_real)
        
        loss = loss_real + loss_fake + 1.0 * loss_gp
        
        return loss, {
            'D/real': loss_real.item(),
            'D/fake': loss_fake.item(),
            'D/gp': loss_gp.item()
        }

    def compute_g_loss(self, x_real, y_org, x_ref, y_trg, x_ref2, dynamic_weights=None):
        """Computes generator and style encoder losses using dynamic weights"""
        losses = {}
        
        weights = {**dynamic_weights, 'gan': self.loss_weights.get('gan', 1.0)} if dynamic_weights else self.loss_weights
        
        s_trg = self.style_encoder(x_ref, y_trg)
        x_fake = self.generator(x_real, s_trg)
        
        if weights.get('style_recon', 0) > 0:
            s_pred = self.style_encoder(x_fake, y_trg)
            losses['style_recon'] = torch.mean(torch.abs(s_pred - s_trg))
        else:
            losses['style_recon'] = torch.tensor(0.0, device=self.device)
        
        if weights.get('style_div', 0) > 0:
            s_trg2 = self.style_encoder(x_ref2, y_trg)
            x_fake2 = self.generator(x_real, s_trg2)
            losses['style_div'] = -torch.mean(torch.abs(x_fake - x_fake2.detach()))
        else:
            losses['style_div'] = torch.tensor(0.0, device=self.device)

        if weights.get('cycle', 0) > 0:
            s_org = self.style_encoder(x_real, y_org)
            x_recon = self.generator(x_fake, s_org)
            losses['cycle'] = torch.mean(torch.abs(x_recon - x_real))
        else:
            losses['cycle'] = torch.tensor(0.0, device=self.device)
        
        if weights.get('identity', 0) > 0:
            s_org = self.style_encoder(x_real, y_org)
            x_idt = self.generator(x_real, s_org)
            losses['identity'] = torch.mean(torch.abs(x_idt - x_real))
        else:
            losses['identity'] = torch.tensor(0.0, device=self.device)
        
        out_fake = self.discriminator(x_fake, y_trg)
        if out_fake.dim() == 1:
            out_fake = out_fake.unsqueeze(-1)
        losses['gan'] = F.binary_cross_entropy_with_logits(out_fake, torch.ones_like(out_fake))
        
        if weights.get('content', 0) > 0 or weights.get('style', 0) > 0:
            content_loss, style_loss = self.criterion_vgg(x_fake, x_ref, x_real)
            losses['content'] = content_loss
            losses['style'] = style_loss
        else:
            losses['content'] = torch.tensor(0.0, device=self.device)
            losses['style'] = torch.tensor(0.0, device=self.device)
        
        total_loss = sum(weights.get(name, 0.0) * loss for name, loss in losses.items() if weights.get(name, 0.0) != 0)
        
        return total_loss, losses

    def gradient_penalty(self, d_out, x_in):
        """Computes the gradient penalty for the discriminator"""
        batch_size = x_in.size(0)
        if d_out.dim() > 1:
            d_out = d_out.mean()
        
        grad_dout = torch.autograd.grad(
            outputs=d_out, inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
    
    def train_d_step(self, batch):
        """A single training step for the discriminator"""
        x_real = batch['source'].to(self.device)
        x_ref = batch['target'].to(self.device)
        y_org = batch['source_domain'].to(self.device)
        y_trg = batch['target_domain'].to(self.device)
        
        self.d_optimizer.zero_grad()
        d_loss, d_losses_log = self.compute_d_loss(x_real, y_org, x_ref, y_trg)
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_losses_log

    def train_g_step(self, batch, dynamic_weights=None):
        """A single training step for the generator"""
        x_real = batch['source'].to(self.device)
        x_ref = batch['target'].to(self.device)
        x_ref2 = batch['target2'].to(self.device)
        y_org = batch['source_domain'].to(self.device)
        y_trg = batch['target_domain'].to(self.device)
        
        self.g_optimizer.zero_grad()
        g_loss, g_losses_raw = self.compute_g_loss(
            x_real, y_org, x_ref, y_trg, x_ref2, 
            dynamic_weights=dynamic_weights
        )
        g_loss.backward()
        self.g_optimizer.step()
        
        self.ema.update_model_average(self.ema_generator, self.generator)
        self.ema.update_model_average(self.ema_style_encoder, self.style_encoder)
        
        g_losses_log = {}
        for name, loss in g_losses_raw.items():
            g_losses_log[f'G/{name}'] = loss.item() if torch.is_tensor(loss) else loss
        g_losses_log['G/total'] = g_loss.item()
        
        return g_losses_log, g_losses_raw

    def save_checkpoint(self, save_dir, epoch):
        """Saves a model checkpoint"""
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
            'weight_history': self.weight_scheduler.weight_history,
            'loss_scheduler_history': self.weight_scheduler.loss_history,
        }
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Loads a model checkpoint"""
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
        
        if 'weight_history' in checkpoint:
            self.weight_scheduler.weight_history = checkpoint['weight_history']
        if 'loss_scheduler_history' in checkpoint:
            self.weight_scheduler.loss_history = checkpoint['loss_scheduler_history']
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch


def train(model, dataset, cfg, start_epoch=0):
    """Main training loop, includes asymmetric updates"""
    save_dir = os.path.join(cfg.save_dir_base, cfg.EXPERIMENT_NAME)
    sample_dir = os.path.join(save_dir, 'samples')
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    fixed_samples = dataset.get_fixed_samples(num_samples=4)
    
    if cfg.wandb:
        wandb.watch(
            models=(model.generator, model.style_encoder, model.discriminator),
            log_freq=100
        )
        
    # --- Added: Set G/D update ratio ---
    g_update_ratio = 2  # Update generator 5 times for every 1 discriminator update
    print(f"Training will proceed with a G:D ratio of {g_update_ratio}:1.")

    d_losses_log_buffer = {} # Used to buffer the D loss logs

    for epoch in range(start_epoch, cfg.epochs):
        epoch_losses = {}
        epoch_g_losses = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
        
        for i, batch in enumerate(pbar):
            if i == 0:
                dynamic_weights = model.weight_scheduler.get_current_weights(
                    epoch, epoch_g_losses if epoch_g_losses else {}
                )
                print(f"\nEpoch {epoch+1} dynamic weights: {dynamic_weights}")
            
            # --- Modified training steps ---
            # 1. Train the generator (executes every iteration)
            g_losses_log, g_losses_raw = model.train_g_step(batch, dynamic_weights)
            
            # 2. Train the discriminator according to the ratio
            if i % g_update_ratio == 0:
                d_losses_log = model.train_d_step(batch)
                d_losses_log_buffer = d_losses_log # Update the D loss log
                
            # Combine logs for display
            current_log = {**g_losses_log, **d_losses_log_buffer}

            for k, v in g_losses_raw.items():
                if k not in epoch_g_losses: epoch_g_losses[k] = []
                epoch_g_losses[k].append(v.item() if torch.is_tensor(v) else v)
            
            pbar.set_postfix({k: f'{v:.3f}' for k, v in current_log.items()})
            
            for k, v in current_log.items():
                if k not in epoch_losses: epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            if cfg.wandb and i % 10 == 0:
                log_dict = {f'step/{k}': v for k, v in current_log.items()}
                for k, v in dynamic_weights.items():
                    log_dict[f'weight/{k}'] = v
                wandb.log(log_dict)
            
            if i % cfg.save_freq == 0:
                with torch.no_grad():
                    all_rows = []
                    num_samples_per_domain = 2
                    
                    # Get a new batch to generate images to avoid using a batch already in training
                    try:
                        vis_batch = next(iter(dataloader))
                    except StopIteration:
                        vis_batch = batch # If the dataloader is exhausted, use the current batch

                    for src_idx in range(min(num_samples_per_domain, vis_batch['source'].size(0))):
                        source = vis_batch['source'][src_idx:src_idx+1].to(model.device)
                        row = [source]
                        
                        for domain_idx in range(1, dataset.num_domains):
                            if domain_idx in fixed_samples['targets'] and len(fixed_samples['targets'][domain_idx]) > 0:
                                ref_idx = i % len(fixed_samples['targets'][domain_idx])
                                ref = fixed_samples['targets'][domain_idx][ref_idx].unsqueeze(0).to(model.device)
                                y_trg = torch.tensor([domain_idx], device=model.device)
                                
                                s = model.ema_style_encoder(ref, y_trg)
                                fake = model.ema_generator(source, s)
                                row.append(fake)
                        
                        if len(row) > 1:
                            all_rows.append(torch.cat(row, dim=0))
                    
                    if all_rows:
                        samples_grid = torch.cat(all_rows, dim=0)
                        save_image(
                            samples_grid,
                            os.path.join(sample_dir, f'epoch_{epoch+1:03d}_iter_{i:05d}.png'),
                            nrow=dataset.num_domains,
                            normalize=True,
                            value_range=(-1, 1)
                        )
        
        avg_g_losses = {k: np.mean(v) for k, v in epoch_g_losses.items()}
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        for k, v in avg_losses.items():
            if k not in model.loss_history: model.loss_history[k] = []
            model.loss_history[k].append(v)
        
        if cfg.wandb:
            epoch_log = {'epoch': epoch + 1}
            epoch_log.update({f'epoch/{k}': v for k, v in avg_losses.items()})
            epoch_log['lr/generator'] = model.g_scheduler.get_last_lr()[0]
            epoch_log['lr/discriminator'] = model.d_scheduler.get_last_lr()[0]
            for k, v in dynamic_weights.items():
                epoch_log[f'epoch_weight/{k}'] = v
            wandb.log(epoch_log)
        
        model.g_scheduler.step()
        model.d_scheduler.step()
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1:03d}')
            model.save_checkpoint(checkpoint_path, epoch)
            
            model.weight_scheduler.plot_weight_history(
                save_path=os.path.join(save_dir, 'weight_history.png')
            )
        
        if len(model.loss_history) > 0:
            plt.figure(figsize=(12, 8))
            for key, values in model.loss_history.items():
                if len(values) > 0: plt.plot(values, label=key)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Losses')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
            plt.close()
    
    model.weight_scheduler.plot_weight_history(
        save_path=os.path.join(save_dir, 'weight_history.png')
    )
    
    print("Training complete!")

