import torch
import argparse
import json
import os

# W&B for experiment tracking
import wandb

# Import default configurations and modules from our new files
import config as default_config
from dataset import MultiDomainStyleTransferDataset
from trainer import MultiDomainStyleCycleGAN, train_multi_domain_style_cyclegan


def main(cfg):
    """
    The main function to set up and run the multi-domain training experiment.
    """
    # --- W&B Initialization ---
    if cfg.wandb:
        # Sanitize config dictionary for W&B
        config_dict = {key: value for key, value in vars(cfg).items() if not key.startswith('__')}
        wandb.init(
            project="Multi-Domain Style-Injected CycleGAN",
            name=cfg.EXPERIMENT_NAME,
            config=config_dict
        )

    # --- Device Setup ---
    if torch.cuda.is_available() and cfg.gpu >= 0:
        device = torch.device(f'cuda:{cfg.gpu}')
        print(f"Using GPU {cfg.gpu}: {torch.cuda.get_device_name(cfg.gpu)}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, using CPU for training.")

    print(f"--- Starting Multi-Domain Experiment: {cfg.EXPERIMENT_NAME} ---")

    # --- Directory Validation ---
    if not os.path.exists(cfg.source_dir):
        print(f"ERROR: Source directory not found: {cfg.source_dir}")
        return
        
    if not os.path.exists(cfg.target_dir):
        print(f"ERROR: Target domains directory not found: {cfg.target_dir}")
        return

    # --- Data Loading ---
    dataset = MultiDomainStyleTransferDataset(cfg.source_dir, cfg.target_dir, cfg.image_size)
    
    # Print dataset statistics
    print(f"\n{'='*60}")
    print("MULTI-DOMAIN DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total domains: {dataset.num_domains}")
    print(f"Source domain: {dataset.domains[0]} (index 0)")
    print(f"Target domains: {dataset.num_target_domains}")
    
    for i, domain_name in enumerate(dataset.domains):
        if i == 0:
            print(f"  • Domain {i}: {domain_name} - {len(dataset.source_files)} images")
        else:
            num_images = len(dataset.target_files_by_domain[domain_name])
            print(f"  • Domain {i}: {domain_name} - {num_images} images")
    print(f"{'='*60}\n")

    # --- Model and Trainer Initialization ---
    model = MultiDomainStyleCycleGAN(
        device=device, 
        total_epochs=cfg.epochs,
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        loss_weights=cfg.LOSS_WEIGHTS,
        num_domains=dataset.num_domains
    )

    # --- Resume Training ---
    start_epoch = 0
    if cfg.resume:
        print(f"Attempting to resume from checkpoint: {cfg.resume}")
        start_epoch = model.load_models(cfg.resume)
        print(f"Resuming training from epoch {start_epoch + 1}")

    # --- Start Training ---
    print("Starting multi-domain training...")
    try:
        train_multi_domain_style_cyclegan(model, dataset, cfg, start_epoch=start_epoch)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- W&B Finalization ---
        if cfg.wandb:
            wandb.finish()
            
    print(f"--- Multi-Domain Experiment {cfg.EXPERIMENT_NAME} Completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Domain StyleCycleGAN with custom configurations.")
    
    # --- Path Arguments ---
    parser.add_argument('--source_dir', type=str, default=default_config.SOURCE_DIR,
                        help='Path to source domain directory')
    parser.add_argument('--target_dir', type=str, default=default_config.TARGET_DIR,
                        help='Path to parent directory containing target domain subdirectories')
    parser.add_argument('--save_dir_base', type=str, default=default_config.SAVE_DIR_BASE,
                        help='Base directory for saving results')
    parser.add_argument('--resume', type=str, default=default_config.RESUME_CHECKPOINT,
                        help='Path to the checkpoint directory to resume training from')

    # --- Experiment Arguments ---
    parser.add_argument('--exp_name', type=str, help='Experiment name. If not provided, it will be auto-generated.')
    parser.add_argument('--gpu', type=int, default=default_config.GPU, help='GPU ID to use (-1 for CPU).')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=default_config.NUM_EPOCHS)
    parser.add_argument('--image_size', type=int, default=default_config.IMAGE_SIZE)
    parser.add_argument('--batch_size', type=int, default=default_config.BATCH_SIZE)
    parser.add_argument('--save_freq', type=int, default=default_config.SAVE_FREQ)
    parser.add_argument('--lr_g', type=float, default=default_config.LEARNING_RATE_G)
    parser.add_argument('--lr_d', type=float, default=default_config.LEARNING_RATE_D)
    parser.add_argument('--loss_weights', type=str, default=json.dumps(default_config.LOSS_WEIGHTS), 
                        help='Loss weights as a JSON string.')
    parser.add_argument('--use_ema', type=bool, default=default_config.TRAINING_USE_EMA, 
                        help='Use EMA models for saving samples during training.')

    # --- W&B Logging ---
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')

    args = parser.parse_args()

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.LOSS_WEIGHTS = json.loads(self.loss_weights)

            if not self.exp_name:
                sorted_weights = sorted(self.LOSS_WEIGHTS.items())
                name_parts = [f"{key}{str(value).replace('.', 'p')}" for key, value in sorted_weights]
                self.exp_name = f"multi_domain_{'_'.join(name_parts)}"            
 
            self.EXPERIMENT_NAME = self.exp_name

    cfg = Config(**vars(args))
    main(cfg)