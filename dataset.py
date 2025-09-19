import os
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiDomainStyleTransferDataset(Dataset):
    """
    Multi-domain dataset for StyleCycleGAN.
    Source domain is always index 0, target domains start from index 1.
    """
    def __init__(self, source_root, target_root, image_size):
        super().__init__()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomChoice([
                transforms.RandomRotation([angle, angle]) for angle in [0, 90, 180, 270]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

        # Load source domain files
        self.source_files = self._get_image_files(source_root)
        print(f"Found {len(self.source_files)} source images")
        
        # Initialize domain mapping - source is always domain 0
        self.domains = ['source']
        self.domain_to_idx = {'source': 0}
        self.target_files_by_domain = {}
        
        # Discover all target domains
        if os.path.isdir(target_root):
            domain_dirs = [d for d in os.listdir(target_root) 
                          if os.path.isdir(os.path.join(target_root, d))]
            domain_dirs = sorted(domain_dirs)
            
            for domain_name in domain_dirs:
                domain_path = os.path.join(target_root, domain_name)
                domain_files = self._get_image_files(domain_path)
                
                if len(domain_files) > 0:
                    idx = len(self.domains)
                    self.domains.append(domain_name)
                    self.domain_to_idx[domain_name] = idx
                    self.target_files_by_domain[domain_name] = domain_files
                    print(f"Domain {idx}: {domain_name} - {len(domain_files)} images")
        
        self.num_domains = len(self.domains)
        self.num_target_domains = len(self.domains) - 1
        
        if self.num_target_domains == 0:
            raise ValueError(f"No target domains found in {target_root}")
            
        print(f"Total domains: {self.num_domains} (1 source + {self.num_target_domains} targets)")
    
    def _get_image_files(self, directory):
        """Get all image files from a directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(files)
    
    def __getitem__(self, index):
        # Load source image
        source_idx = index % len(self.source_files)
        source_img = Image.open(self.source_files[source_idx]).convert('RGB')
        source_img = self.transform(source_img)
        
        # Randomly select a target domain
        target_domain_names = list(self.target_files_by_domain.keys())
        domain_name = random.choice(target_domain_names)
        target_domain_idx = self.domain_to_idx[domain_name]
        domain_files = self.target_files_by_domain[domain_name]
        
        # Sample target image
        target_file = random.choice(domain_files)
        target_img = Image.open(target_file).convert('RGB')
        target_img = self.transform(target_img)
        
        return {
            'source': source_img,
            'target': target_img,
            'source_domain': 0,  # Source is always domain 0
            'target_domain': target_domain_idx
        }

    def __len__(self):
        return max(len(self.source_files), 
                  max(len(files) for files in self.target_files_by_domain.values()))

class InferenceDataset(Dataset):
    """
    Dataset for inference - loads images from a single directory.
    """
    def __init__(self, input_dir, image_size):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
        
        # Get all image files from input directory
        self.image_files = self._get_image_files(input_dir)
        print(f"Found {len(self.image_files)} images for inference in {input_dir}")
    
    def _get_image_files(self, directory):
        """Get all image files from a directory."""
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return []
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(files)
    
    def __getitem__(self, index):
        file_path = self.image_files[index]
        image = Image.open(file_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Return tensor and filename
        filename = os.path.basename(file_path)
        return image_tensor, filename
    
    def __len__(self):
        return len(self.image_files)
