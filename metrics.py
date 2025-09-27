import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import os
from tqdm import tqdm
import datetime
import config
import traceback

class ImageFolder(Dataset):
    def __init__(self, root_dir, for_metrics=True):
        self.root_dir = root_dir
        self.image_files = []

        # Fixed: Use lowercase extensions since we use .lower() below
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for root, _, files in os.walk(root_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    self.image_files.append(os.path.join(root, filename))
        
        if not self.image_files:
            raise ValueError(f"No valid images found in {root_dir}")
            
        print(f"Loading {len(self.image_files)} images from {root_dir}")
        
        if for_metrics:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy tensor and path to avoid crashing the whole batch
            return torch.zeros((3, 299, 299), dtype=torch.uint8)

def calculate_metrics(folder_dir, target_dir, device):
    """Calculate FID and KID metrics between a folder and target images"""
    try:
        folder_dataset = ImageFolder(folder_dir, for_metrics=True)
        target_dataset = ImageFolder(target_dir, for_metrics=True)
        
        folder_loader = DataLoader(folder_dataset, batch_size=32, 
                                 num_workers=4, shuffle=False, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=32, 
                                 num_workers=4, shuffle=False, pin_memory=True)
        
        # Initialize metrics
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        
        # 計算適當的 subset_size（取兩個數據集中較小的size的一半）
        min_samples = min(len(folder_dataset), len(target_dataset))
        subset_size = max(min(50, min_samples // 2), 2)  # 至少2張，最多50張
        print(f"Using KID subset_size of {subset_size} (total samples: {len(folder_dataset)}/{len(target_dataset)})")
        kid_metric = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
        
        # === MODIFICATION START: Use batch-wise updates for memory efficiency ===
        print("Processing generated images...")
        for batch in tqdm(folder_loader, desc="Generated images"):
            images = batch.to(device)
            fid_metric.update(images, real=False)
            kid_metric.update(images, real=False)

        print("Processing real/target images...")
        for batch in tqdm(target_loader, desc="Target images"):
            images = batch.to(device)
            fid_metric.update(images, real=True)
            kid_metric.update(images, real=True)
            
        print("Calculating final scores...")
        fid_score = float(fid_metric.compute())
        kid_mean, _ = kid_metric.compute()
        # === MODIFICATION END ===
        
        return {
            'FID': fid_score,
            'KID': float(kid_mean) * 1000,
            'folder_count': len(folder_dataset),
            'target_count': len(target_dataset)
        }
        
    except Exception as e:
        print(f"Error during calculation for {folder_dir}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    # Multi-domain experiment directory structure
    base_dir = config.METRICS_INPUT_DIR  # './results/multi_domain_exp/output'
    target_base_dir = config.METRICS_TARGET_DIR  # './data/all'

    output_filename = f"metrics_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(base_dir, output_filename)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get all epoch directories (support both "epoch_X" and "X" formats)
    epoch_dirs = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)):
            if d.startswith('epoch_') and d[6:].isdigit():
                epoch_dirs.append(d)
            elif d.isdigit():
                epoch_dirs.append(d)
    
    if not epoch_dirs:
        print(f"No epoch directories found in {base_dir}")
        return
    
    # Sort epoch directories numerically
    def extract_epoch_number(dirname):
        if dirname.startswith('epoch_'):
            return int(dirname[6:])
        else:
            return int(dirname)
    
    epoch_dirs.sort(key=extract_epoch_number)
    
    # Calculate metrics for each epoch and domain combination
    results = {}
    print(f"\nFound epochs: {epoch_dirs}")
    print("Calculating metrics for all combinations...")
    
    for epoch_dir in epoch_dirs:
        epoch_path = os.path.join(base_dir, epoch_dir)
        
        # Get all domain directories in this epoch
        domain_dirs = [d for d in os.listdir(epoch_path) 
                      if os.path.isdir(os.path.join(epoch_path, d))]
        
        if not domain_dirs:
            print(f"No domain directories found in epoch {epoch_dir}")
            continue
            
        print(f"\nProcessing epoch {epoch_dir} with domains: {domain_dirs}")
        
        for domain_name in domain_dirs:
            # Input directory: {base_dir}/{epoch_n}/{domain_name}/interpolate
            input_dir = os.path.join(epoch_path, domain_name, 'interpolate')
            
            # Target directory: {target_base_dir}/{domain_name}
            target_dir = os.path.join(target_base_dir, domain_name)
            
            # Check if both directories exist
            if not os.path.exists(input_dir):
                print(f"Warning: Input directory does not exist: {input_dir}")
                continue
                
            if not os.path.exists(target_dir):
                print(f"Warning: Target directory does not exist: {target_dir}")
                continue
            
            # Extract epoch number for display
            display_epoch = epoch_dir[6:] if epoch_dir.startswith('epoch_') else epoch_dir
            print(f"\nProcessing epoch {display_epoch}, domain {domain_name}...")
            print(f"Input: {input_dir}")
            print(f"Target: {target_dir}")
            
            metrics = calculate_metrics(input_dir, target_dir, device)
            if metrics:
                # Extract epoch number for display (remove "epoch_" prefix if present)
                display_epoch = epoch_dir[6:] if epoch_dir.startswith('epoch_') else epoch_dir
                
                # Use epoch_domain as key for better organization
                key = f"epoch_{display_epoch}_{domain_name}"
                results[key] = {
                    'epoch': display_epoch,
                    'domain': domain_name,
                    **metrics
                }
    
    if not results:
        print("No valid results found!")
        return
    
    # Calculate dynamic column widths for better formatting
    max_domain_length = max(len(results[key]['domain']) for key in results.keys()) if results else 20
    domain_width = max(max_domain_length, 20)  # At least 20 characters for "Domain" header
    
    # Print and save results
    output_text = []
    # Add timestamp and directory information
    output_text.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_text.append(f"Base directory: {os.path.abspath(base_dir)}")
    output_text.append(f"Target base directory: {os.path.abspath(target_base_dir)}")
    output_text.append("\nResults:")
    
    # Dynamic header with proper spacing
    header = f"{'Epoch':<8} {'Domain':<{domain_width}} {'FID ↓':>8} {'KID (x10³) ↓':>12} {'Images':>10}"
    separator = "-" * len(header)
    
    output_text.append(header)
    output_text.append(separator)
    
    # Track best performing combinations for each domain
    domain_best_fid = {}
    domain_best_kid = {}
    
    # Sort results by epoch then domain for better readability
    sorted_keys = sorted(results.keys(), key=lambda x: (int(results[x]['epoch']), results[x]['domain']))
    
    for key in sorted_keys:
        metrics = results[key]
        epoch = metrics['epoch']
        domain = metrics['domain']
        
        result_line = f"{epoch:<8} {domain:<{domain_width}} {metrics['FID']:8.2f} {metrics['KID']:12.2f} {metrics['folder_count']:>4}/{metrics['target_count']:<4}"
        print(result_line)
        output_text.append(result_line)
        
        # Track best performing epoch for each domain
        if domain not in domain_best_fid or metrics['FID'] < domain_best_fid[domain]['FID']:
            domain_best_fid[domain] = {'epoch': epoch, 'FID': metrics['FID']}
        if domain not in domain_best_kid or metrics['KID'] < domain_best_kid[domain]['KID']:
            domain_best_kid[domain] = {'epoch': epoch, 'KID': metrics['KID']}
    
    # Add best performing information with dynamic formatting
    output_text.append("\nBest performing epochs for each domain:")
    best_header = f"{'Domain':<{domain_width}} {'Best FID (Epoch)':<18} {'Best KID (Epoch)':<18}"
    best_separator = "-" * len(best_header)
    
    output_text.append(best_header)
    output_text.append(best_separator)
    
    all_domains = set(domain_best_fid.keys()) | set(domain_best_kid.keys())
    for domain in sorted(all_domains):
        fid_info = domain_best_fid.get(domain, {'epoch': 'N/A', 'FID': 'N/A'})
        kid_info = domain_best_kid.get(domain, {'epoch': 'N/A', 'KID': 'N/A'})
        
        if isinstance(fid_info['FID'], float):
            fid_str = f"{fid_info['FID']:.2f} ({fid_info['epoch']})"
        else:
            fid_str = f"{fid_info['FID']} ({fid_info['epoch']})"
            
        if isinstance(kid_info['KID'], float):
            kid_str = f"{kid_info['KID']:.2f} ({kid_info['epoch']})"
        else:
            kid_str = f"{kid_info['KID']} ({kid_info['epoch']})"
        
        best_line = f"{domain:<{domain_width}} {fid_str:<18} {kid_str:<18}"
        print(best_line)
        output_text.append(best_line)
    
    # Create output directory if needed
    output_dir_path = os.path.dirname(output_path)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_text))
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()