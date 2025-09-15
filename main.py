import torch
import argparse
import json
import os
import wandb

# Import modules
import config as default_config
from dataset import MultiDomainDataset
from trainer import MSIGAN, train
from utils import setup_logger


def main(args):
    """Main function to setup and run multi-domain training."""
    
    # Setup logger
    logger = setup_logger(os.path.join(args.save_dir_base, args.exp_name))
    logger.info(f"Starting experiment: {args.exp_name}")
    
    # W&B initialization
    if args.wandb:
        config_dict = {key: value for key, value in vars(args).items() 
                      if not key.startswith('__')}
        wandb.init(
            project="Multi-Domain Style-Injected GAN",
            name=args.exp_name,
            config=config_dict
        )
        logger.info("W&B tracking enabled")
    
    # Device setup
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    # Validate directories
    if not os.path.exists(args.source_dir):
        raise ValueError(f"Source directory not found: {args.source_dir}")
    if not os.path.exists(args.target_dir):
        raise ValueError(f"Target directory not found: {args.target_dir}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = MultiDomainDataset(
        source_root=args.source_dir,
        target_root=args.target_dir,
        image_size=args.image_size,
        mode='train'
    )
    
    num_domains = dataset.num_domains  # This now includes source domain
    logger.info(f"Dataset loaded with {num_domains} total domains (1 source + {dataset.num_target_domains} targets)")
    
    # Create model
    logger.info("Building model...")
    model = MSIGAN(
        img_size=args.image_size,
        style_dim=args.style_dim,
        num_domains=num_domains,  # Total domains including source
        device=device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        loss_weights=args.loss_weights,
        total_epochs=args.epochs
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.generator.parameters())
    logger.info(f"Generator parameters: {total_params:,}")
    total_params = sum(p.numel() for p in model.style_encoder.parameters())
    logger.info(f"Style Encoder parameters: {total_params:,}")
    total_params = sum(p.numel() for p in model.discriminator.parameters())
    logger.info(f"Discriminator parameters: {total_params:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_path = os.path.join(args.resume, 'checkpoint.pth')
        start_epoch = model.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create configuration object for trainer
    class TrainingConfig:
        def __init__(self, args):
            self.save_dir_base = args.save_dir_base
            self.EXPERIMENT_NAME = args.exp_name
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.save_freq = args.save_freq
            self.NUM_WORKERS = args.num_workers
            self.wandb = args.wandb
    
    cfg = TrainingConfig(args)
    
    # Start training
    logger.info("Starting training...")
    try:
        train(model, dataset, cfg, start_epoch=start_epoch)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
    finally:
        if args.wandb:
            wandb.finish()
            logger.info("W&B run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Multi-Domain Style Translation Model"
    )
    
    # Paths
    parser.add_argument('--source_dir', type=str, default=default_config.SOURCE_DIR,
                        help='Path to source domain directory')
    parser.add_argument('--target_dir', type=str, default=default_config.TARGET_DOMAINS_DIR,
                        help='Path to parent directory containing target domains')
    parser.add_argument('--save_dir_base', type=str, default=default_config.SAVE_DIR_BASE,
                        help='Base directory for saving results')
    parser.add_argument('--resume', type=str, default=default_config.RESUME_CHECKPOINT,
                        help='Path to checkpoint directory to resume from')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default=default_config.EXPERIMENT_NAME,
                        help='Experiment name')
    parser.add_argument('--gpu', type=int, default=default_config.GPU,
                        help='GPU device ID (-1 for CPU)')
    
    # Model architecture
    parser.add_argument('--image_size', type=int, default=default_config.IMAGE_SIZE,
                        help='Image size')
    parser.add_argument('--style_dim', type=int, default=default_config.STYLE_DIM,
                        help='Dimension of style code')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=default_config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=default_config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=default_config.LEARNING_RATE_G,
                        help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=default_config.LEARNING_RATE_D,
                        help='Learning rate for discriminator')
    parser.add_argument('--save_freq', type=int, default=default_config.SAVE_FREQ,
                        help='Frequency of saving samples')
    parser.add_argument('--num_workers', type=int, default=default_config.NUM_WORKERS,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--loss_weights', type=json.loads, 
                        default=json.dumps(default_config.LOSS_WEIGHTS),
                        help='Loss weights as JSON string')
    
    # Options
    parser.add_argument('--use_ema', action='store_true', 
                        default=default_config.TRAINING_USE_EMA,
                        help='Use EMA for model averaging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    
    args = parser.parse_args()
    
    # Parse loss weights if provided as string
    if isinstance(args.loss_weights, str):
        args.loss_weights = json.loads(args.loss_weights)
    
    main(args)