import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
from tqdm import tqdm
import glob
import random
import re

import config as default_config
from model import build_model
from dataset import InferenceDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional VAE support - keeping function name from Style-Injected-CycleGAN
try:
    from Style_Code_VAE.train_style_code_vae import StyleCodeVAE
except ImportError:
    print("Warning: Could not import StyleCodeVAE. 'vae' mode will be unavailable.")
    StyleCodeVAE = None


def parse_vae_params_from_filename(checkpoint_path):
    """
    Parse beta and latent_dim from the VAE checkpoint filename.
    Expected format: stylecodevae_beta{X.X}_ld{Y}.pth
    
    Returns:
        tuple: (beta, latent_dim) or (None, None) if parsing fails
    """
    filename = os.path.basename(checkpoint_path)
    
    # Pattern to match: stylecodevae_beta{number}_ld{number}.pth
    pattern = r'stylecodevae_beta([0-9.]+)_ld([0-9]+)\.pth'
    match = re.match(pattern, filename)
    
    if match:
        beta = float(match.group(1))
        latent_dim = int(match.group(2))
        return beta, latent_dim
    else:
        print(f"Warning: Could not parse beta and latent_dim from filename: {filename}")
        return None, None


def load_style_code_vae_model(args, device):
    """
    Loads the trained StyleCodeVAE model with automatic parameter detection.
    Keeping function name from Style-Injected-CycleGAN.
    """
    if StyleCodeVAE is None:
        print("Error: StyleCodeVAE class was not imported, cannot load the model.")
        return None

    checkpoint_path = args.vae_checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: VAE checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading StyleCodeVAE model weights from: {checkpoint_path}")
    
    # Try to parse parameters from filename first
    beta_from_file, latent_dim_from_file = parse_vae_params_from_filename(checkpoint_path)
    
    # Use parameters from filename if available, otherwise fall back to config/args
    if latent_dim_from_file is not None:
        latent_dim = latent_dim_from_file
        print(f"Detected latent_dim from filename: {latent_dim}")
    else:
        latent_dim = args.vae_latent_dim
        print(f"Using latent_dim from config/args: {latent_dim}")
    
    # Initialize the VAE model with the correct dimensions
    model = StyleCodeVAE(
        style_dim=args.style_dim,
        latent_dim=latent_dim
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        print("StyleCodeVAE model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the StyleCodeVAE model weights: {e}")
        
        # If automatic detection failed, suggest the correct parameters
        if latent_dim_from_file is not None and latent_dim_from_file != args.vae_latent_dim:
            print(f"Suggestion: The checkpoint appears to be trained with latent_dim={latent_dim_from_file}, ")
            print(f"but the current configuration uses latent_dim={args.vae_latent_dim}.")
            print(f"Please update INFERENCE_VAE_LATENT_DIM in config.py to {latent_dim_from_file}")
        
        return None


def load_model(checkpoint_path, img_size, style_dim, num_domains, device):
    """Load trained multi-domain model."""
    # Build model
    generator, style_encoder, _ = build_model(
        img_size, style_dim, num_domains, device
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load EMA models if available, otherwise use regular models
    if 'ema_generator' in checkpoint:
        generator.load_state_dict(checkpoint['ema_generator'])
        style_encoder.load_state_dict(checkpoint['ema_style_encoder'])
        print("Loaded EMA models")
    else:
        generator.load_state_dict(checkpoint['generator'])
        style_encoder.load_state_dict(checkpoint['style_encoder'])
        print("Loaded regular models")
    
    generator.eval()
    style_encoder.eval()
    
    return generator, style_encoder


def preload_style_vectors(args, style_encoder, target_domain_path, target_domain_idx, device):
    """
    Preload and return all style codes from a directory of target domain images.
    Adapted function name structure from Style-Injected-CycleGAN but with multi-domain logic.
    """
    if not os.path.isdir(target_domain_path):
        print(f"ERROR: '{target_domain_path}' is not a valid target domain path.")
        return None

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    style_files = []
    for ext in image_extensions:
        style_files.extend(glob.glob(os.path.join(target_domain_path, ext)))
    
    if not style_files:
        print(f"Style images not found in '{target_domain_path}'.")
        return None
    
    # Limit number of styles if specified
    if args.max_styles and len(style_files) > args.max_styles:
        style_files = random.sample(style_files, args.max_styles)
    
    print(f"Found {len(style_files)} style images. Preloading all style vectors...")
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    
    # Extract style vectors
    all_style_vectors = []
    with torch.no_grad():
        for style_path in tqdm(style_files, desc="Encoding all styles..."):
            style_image = Image.open(style_path).convert('RGB')
            style_tensor = transform(style_image).unsqueeze(0).to(device)
            
            # Create domain label tensor for multi-domain
            y = torch.tensor([target_domain_idx], device=device)
            
            # Extract style using multi-domain style encoder
            style_code = style_encoder(style_tensor, y)
            all_style_vectors.append(style_code)
    
    if not all_style_vectors:
        print("Failure to extract any style encoding from the target domain image.")
        return None
    
    print(f"Successfully preloaded {len(all_style_vectors)} style vectors.")
    return all_style_vectors


def apply_style_mode(style_vectors, mode, noise_level=0.1, vae_model=None, device=None, args=None):
    """
    Apply different style sampling strategies.
    Keeping the core logic from both implementations.
    """
    if not style_vectors and mode != 'vae':
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
        
    elif mode == 'vae':
        if vae_model is None:
            raise ValueError("VAE model required for 'vae' mode")
        # Get the actual latent dimension from the loaded VAE model
        actual_latent_dim = vae_model.latent_dim
        # Step 1: Sample a random vector z from the VAE's latent space
        z = torch.randn(1, actual_latent_dim).to(device)
        # Step 2: Use the VAE's decoder to generate a new, synthetic style code
        new_style_code = vae_model.decode(z)
        # Step 3: For multi-domain, ensure proper shape
        style = new_style_code
        
    else:
        raise ValueError(f"Unknown style mode: {mode}")
    
    return style


def generate_comparison_grid(dataset, generator, style_vectors, 
                            target_domain_idx, device, output_dir):
    """Generate a grid showing different style modes."""
    # Select a few source images
    num_samples = min(4, len(dataset))
    modes = ['average', 'random', 'interpolate', 'noise', 'vae']
    
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
    generator, style_encoder = load_model(
        checkpoint_path, 
        args.image_size, 
        args.style_dim,
        num_domains,
        device
    )
    
    # Load VAE model if needed - using Style-Injected-CycleGAN function
    vae_model = None
    if args.style_mode == 'vae':
        vae_model = load_style_code_vae_model(args, device)
        if vae_model is None:
            print("Could not load VAE model, terminating program.")
            return
    
    # Preload style vectors from target domain (unless using VAE)
    all_style_vectors = None
    if args.style_mode != 'vae':
        all_style_vectors = preload_style_vectors(
            args, style_encoder, target_domain_path, target_domain_idx, device
        )
        if not all_style_vectors:
            print("Could not load style vectors, terminating program.")
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
        fixed_style = apply_style_mode(all_style_vectors, 'average')
    
    # Process images - following Style-Injected-CycleGAN structure
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Applying style via '{args.style_mode}' mode"):
            img_tensor, img_name = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Dynamic style code generation - following Style-Injected-CycleGAN pattern
            if args.style_mode == 'average':
                current_style_code = fixed_style
            elif args.style_mode == 'random':
                current_style_code = random.choice(all_style_vectors)
            elif args.style_mode == 'interpolate':
                s_A, s_B = random.sample(all_style_vectors, 2)
                alpha = random.random()
                current_style_code = alpha * s_A + (1.0 - alpha) * s_B
            elif args.style_mode == 'noise':
                s = random.choice(all_style_vectors)
                noise = torch.randn_like(s) * args.noise_level
                current_style_code = s + noise
            elif args.style_mode == 'vae':
                # Get the actual latent dimension from the loaded VAE model
                actual_latent_dim = vae_model.latent_dim
                # Sample a random vector z from the VAE's latent space
                z = torch.randn(1, actual_latent_dim).to(device)
                # Use the VAE's decoder to generate a new, synthetic style code
                current_style_code = vae_model.decode(z)
            else:
                raise ValueError(f"Unknown style mode: {args.style_mode}")
            
            # Generate translation
            fake = generator(img_tensor, current_style_code)
            
            # Save result
            output_path = os.path.join(output_subdir, img_name)
            save_image(
                fake,
                output_path,
                normalize=True,
                value_range=(-1, 1)
            )
    
    print(f"\nInference complete! Results saved to: {output_subdir}")
    
    # Generate comparison grid if requested
    if args.save_grid:
        print("Generating comparison grid...")
        generate_comparison_grid(
            dataset, generator, all_style_vectors, 
            target_domain_idx, device, output_subdir
        )


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
    
    # Style sampling - following Style-Injected-CycleGAN choices
    parser.add_argument('--style_mode', type=str, default=default_config.INFERENCE_STYLE_MODE,
                        choices=['average', 'random', 'interpolate', 'noise', 'vae'],
                        help='Style sampling mode')
    parser.add_argument('--noise_level', type=float, default=default_config.INFERENCE_NOISE_LEVEL,
                        help='Noise level for noise mode')
    parser.add_argument('--max_styles', type=int, default=None,
                        help='Maximum number of style vectors to load (None for all)')
    
    # VAE Parameters (only used if style_mode is 'vae') - from Style-Injected-CycleGAN
    parser.add_argument('--vae_checkpoint_path', type=str, 
                        default=getattr(default_config, 'INFERENCE_VAE_CHECKPOINT', ''), 
                        help='Path to the trained StyleCodeVAE checkpoint.')
    parser.add_argument('--vae_latent_dim', type=int, 
                        default=getattr(default_config, 'INFERENCE_VAE_LATENT_DIM', 16), 
                        help='Latent dimension of the StyleCodeVAE model.')
    
    # Options
    parser.add_argument('--save_grid', action='store_true',
                        help='Save comparison grid of different style modes')
    
    args = parser.parse_args()
    main(args)