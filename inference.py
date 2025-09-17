import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
from tqdm import tqdm
import glob
import random

import config as default_config
# FIX 1: Import the correct models instead of build_model
from model import StyleCycleGANGenerator, MultiDomainStyleEncoder
from dataset import InferenceDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(checkpoint_path, img_size, style_dim, num_domains, device):
    """Load trained multi-domain model."""
    print(f"Loading multi-domain model with {num_domains} domains...")
    
    # FIX 2: Build models using the correct architecture
    generator = StyleCycleGANGenerator(
        in_channels=3, 
        out_channels=3, 
        style_dim=style_dim
    ).to(device)
    
    style_encoder = MultiDomainStyleEncoder(
        style_dim=style_dim, 
        num_domains=num_domains
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # FIX 3: Load using the correct checkpoint keys
    try:
        # Try EMA models first
        if 'ema_G_A2B' in checkpoint:
            generator.load_state_dict(checkpoint['ema_G_A2B'])
            style_encoder.load_state_dict(checkpoint['ema_SE_B'])  # Use SE_B for target domain
            print("Loaded EMA models")
        elif 'G_A2B' in checkpoint:
            generator.load_state_dict(checkpoint['G_A2B'])
            style_encoder.load_state_dict(checkpoint['SE_B'])  # Use SE_B for target domain
            print("Loaded regular models")
        else:
            print(f"Available keys: {list(checkpoint.keys())}")
            raise KeyError("Could not find expected model keys in checkpoint")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure the checkpoint was saved with the multi-domain trainer")
        raise
    
    generator.eval()
    style_encoder.eval()
    
    return generator, style_encoder


def preload_style_vectors(style_encoder, ref_domain_dir, domain_idx, 
                         image_size, device, max_styles=None):
    """
    Preload style vectors from a reference domain directory.
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    style_files = []
    for ext in image_extensions:
        style_files.extend(glob.glob(os.path.join(ref_domain_dir, ext)))
    
    if not style_files:
        raise ValueError(f"No images found in {ref_domain_dir}")
    
    # Limit number of styles if specified
    if max_styles and len(style_files) > max_styles:
        style_files = random.sample(style_files, max_styles)
    
    print(f"Loading {len(style_files)} style vectors from {ref_domain_dir}")
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    
    # Extract style vectors
    style_vectors = []
    with torch.no_grad():
        for style_path in tqdm(style_files, desc="Extracting styles"):
            img = Image.open(style_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # FIX 4: Create domain label tensor for multi-domain encoder
            y = torch.tensor([domain_idx], device=device)
            
            # Extract style using multi-domain encoder
            style_code = style_encoder(img_tensor, y)
            style_vectors.append(style_code)
    
    print(f"Loaded {len(style_vectors)} style vectors")
    return style_vectors


def apply_style_mode(style_vectors, mode, noise_level=0.1):
    """
    Apply different style sampling strategies.
    """
    if not style_vectors:
        raise ValueError("No style vectors provided")
    
    if mode == 'average':
        # Average all style vectors
        style = torch.mean(torch.stack(style_vectors), dim=0)
        
    elif mode == 'random':
        # Random selection
        style = random.choice(style_vectors)
        
    elif mode == 'interpolate':
        # Interpolate between two random styles
        if len(style_vectors) < 2:
            style = style_vectors[0]
        else:
            s1, s2 = random.sample(style_vectors, 2)
            alpha = random.random()
            style = alpha * s1 + (1 - alpha) * s2
            
    elif mode == 'noise':
        # Add noise to random style
        style = random.choice(style_vectors)
        noise = torch.randn_like(style) * noise_level
        style = style + noise
        
    elif mode == 'specific':
        # Use first style (consistent)
        style = style_vectors[0]
        
    else:
        raise ValueError(f"Unknown style mode: {mode}")
    
    return style


def main(args):
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Discover domains (source is domain 0, targets start from 1)
    domain_dirs = [d for d in os.listdir(args.ref_domains_dir) 
                   if os.path.isdir(os.path.join(args.ref_domains_dir, d))]
    domain_dirs = sorted(domain_dirs)
    num_target_domains = len(domain_dirs)
    num_domains = num_target_domains + 1  # +1 for source domain
    
    if num_target_domains == 0:
        raise ValueError(f"No domains found in {args.ref_domains_dir}")
    
    print(f"Found {num_target_domains} target domains: {domain_dirs}")
    
    # Find target domain index (add 1 because source is 0)
    if args.target_domain not in domain_dirs:
        raise ValueError(f"Target domain '{args.target_domain}' not found. Available: {domain_dirs}")
    
    target_domain_idx = domain_dirs.index(args.target_domain) + 1  # +1 because source is 0
    target_domain_path = os.path.join(args.ref_domains_dir, args.target_domain)
    
    print(f"Target domain: {args.target_domain} (index: {target_domain_idx})")
    
    # Load model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
    try:
        generator, style_encoder = load_model(
            checkpoint_path, 
            args.image_size, 
            args.style_dim,
            num_domains,
            device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Preload style vectors from target domain
    try:
        style_vectors = preload_style_vectors(
            style_encoder,
            target_domain_path,
            target_domain_idx,
            args.image_size,
            device,
            max_styles=args.max_styles
        )
    except Exception as e:
        print(f"Failed to load style vectors: {e}")
        return
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_subdir = os.path.join(args.output_dir, f'{args.target_domain}_{args.style_mode}')
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load source images
    dataset = InferenceDataset(args.input_dir, args.image_size)
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.input_dir}")
    
    print(f"Processing {len(dataset)} images with style mode: {args.style_mode}")
    
    # Fixed style for 'average' mode
    fixed_style = None
    if args.style_mode == 'average':
        fixed_style = apply_style_mode(style_vectors, 'average')
    
    # Process images
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Generating translations"):
            try:
                img_tensor, img_name = dataset[idx]
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Get style based on mode
                if fixed_style is not None:
                    style = fixed_style
                else:
                    style = apply_style_mode(
                        style_vectors, 
                        args.style_mode,
                        args.noise_level
                    )
                
                # Generate translation
                fake = generator(img_tensor, style)
                
                # Save result
                output_path = os.path.join(output_subdir, img_name)
                save_image(
                    fake,
                    output_path,
                    normalize=True,
                    value_range=(-1, 1)
                )
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    print(f"\nInference complete! Results saved to: {output_subdir}")


def generate_comparison_grid(dataset, generator, style_vectors, 
                            domain_idx, device, output_dir):
    """Generate a grid showing different style modes."""
    # Select a few source images
    num_samples = min(4, len(dataset))
    modes = ['specific', 'average', 'random', 'interpolate', 'noise']
    
    grid = []
    
    for i in range(num_samples):
        img_tensor, _ = dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Add source image
        row = [img_tensor]
        
        # Generate with different modes
        for mode in modes:
            style = apply_style_mode(style_vectors, mode, noise_level=0.1)
            with torch.no_grad():
                fake = generator(img_tensor, style)
            row.append(fake)
        
        grid.append(torch.cat(row, dim=0))
    
    # Save grid
    grid = torch.cat(grid, dim=0)
    save_image(
        grid,
        os.path.join(output_dir, 'style_modes_comparison.png'),
        nrow=len(modes) + 1,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"Comparison grid saved to {output_dir}/style_modes_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-domain inference with style sampling')
    
    # Paths
    parser.add_argument('--input_dir', type=str, default=default_config.INFERENCE_INPUT_DIR,
                        help='Directory containing source images')
    parser.add_argument('--ref_domains_dir', type=str, default=default_config.INFERENCE_REF_DOMAINS_DIR,
                        help='Directory containing all reference domain folders')
    parser.add_argument('--checkpoint_dir', type=str, default=default_config.INFERENCE_CHECKPOINT_DIR,
                        help='Directory containing model checkpoint')
    parser.add_argument('--output_dir', type=str, default=default_config.INFERENCE_OUTPUT_DIR,
                        help='Directory to save output images')
    
    # Domain selection
    parser.add_argument('--target_domain', type=str, default=default_config.INFERENCE_TARGET_DOMAIN,
                        help='Name of target domain folder to translate to')
    
    # Model parameters
    parser.add_argument('--gpu', type=int, default=default_config.GPU,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--image_size', type=int, default=default_config.IMAGE_SIZE,
                        help='Image size')
    parser.add_argument('--style_dim', type=int, default=default_config.STYLE_DIM,
                        help='Dimension of style code')
    
    # Style sampling
    parser.add_argument('--style_mode', type=str, default=default_config.INFERENCE_STYLE_MODE,
                        choices=['average', 'random', 'interpolate', 'noise', 'specific'],
                        help='Style sampling mode')
    parser.add_argument('--noise_level', type=float, default=default_config.INFERENCE_NOISE_LEVEL,
                        help='Noise level for noise mode')
    parser.add_argument('--max_styles', type=int, default=None,
                        help='Maximum number of style vectors to load (None for all)')
    
    # Options
    parser.add_argument('--save_grid', action='store_true',
                        help='Save comparison grid of different style modes')
    
    args = parser.parse_args()
    main(args)