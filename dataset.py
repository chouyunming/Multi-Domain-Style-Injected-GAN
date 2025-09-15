import os
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiDomainDataset(Dataset):
    """
    Dataset for loading source domain and multiple target domains for multi-domain translation.
    FIXED: Proper domain indexing where source=0, targets start from 1
    """
    def __init__(self, source_root, target_root, image_size, mode='train'):
        super().__init__()
        self.image_size = image_size
        self.mode = mode
        
        # Setup transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3)
            ])
        
        # Load source domain files
        self.source_files = self._get_image_files(source_root)
        print(f"Found {len(self.source_files)} source images in {source_root}")
        
        # Load target domains
        self.domains = ['source']  # Domain 0 is source
        self.domain_to_idx = {'source': 0}
        self.target_files_by_domain = {}
        
        # Discover all target domains (they will be indexed from 1 onwards)
        domain_dirs = [d for d in os.listdir(target_root) 
                      if os.path.isdir(os.path.join(target_root, d))]
        domain_dirs = sorted(domain_dirs)  # Ensure consistent ordering
        
        for domain_name in domain_dirs:
            domain_path = os.path.join(target_root, domain_name)
            domain_files = self._get_image_files(domain_path)
            
            if len(domain_files) > 0:
                idx = len(self.domains)  # This will be 1, 2, 3, etc.
                self.domains.append(domain_name)
                self.domain_to_idx[domain_name] = idx
                self.target_files_by_domain[domain_name] = domain_files
                print(f"Domain {idx}: {domain_name} - {len(domain_files)} images")
        
        self.num_domains = len(self.domains)  # Total including source
        self.num_target_domains = len(self.domains) - 1  # Excluding source
        print(f"Total domains: {self.num_domains} (1 source + {self.num_target_domains} targets)")
        
        if self.num_target_domains == 0:
            raise ValueError(f"No target domains found in {target_root}")
        
        # Create a flat list of all target files with their domain labels
        self.all_target_files = []
        self.all_target_labels = []
        for domain_name, files in self.target_files_by_domain.items():
            domain_idx = self.domain_to_idx[domain_name]
            self.all_target_files.extend(files)
            self.all_target_labels.extend([domain_idx] * len(files))
    
    def _get_image_files(self, directory):
        """Get all image files from a directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(files)
    
    def __getitem__(self, index):
        """
        Returns:
            source_img: Image from source domain
            target_img: Image from a random target domain (reference 1)
            target_img2: Another image from the same target domain (reference 2)
            source_domain: Always 0 (source domain index)
            target_domain: Domain index of the target images (1, 2, 3, ...)
        """
        # Load source image
        source_idx = index % len(self.source_files)
        source_img = Image.open(self.source_files[source_idx]).convert('RGB')
        source_img = self.transform(source_img)
        
        # Randomly select a target domain (excluding source)
        target_domain_names = list(self.target_files_by_domain.keys())
        domain_name = random.choice(target_domain_names)
        target_domain = self.domain_to_idx[domain_name]
        domain_files = self.target_files_by_domain[domain_name]
        
        # Sample two different images from the same domain
        if len(domain_files) >= 2:
            # Sample two different images
            selected_files = random.sample(domain_files, 2)
            target_file1, target_file2 = selected_files
        else:
            # If domain has only one image, use it twice (with different augmentation)
            target_file1 = target_file2 = domain_files[0]
        
        # Load and transform both reference images
        target_img = Image.open(target_file1).convert('RGB')
        target_img = self.transform(target_img)
        
        target_img2 = Image.open(target_file2).convert('RGB')
        target_img2 = self.transform(target_img2)
        
        return {
            'source': source_img,
            'target': target_img,
            'target2': target_img2,  # Second reference from same domain
            'source_domain': 0,  # Source is always domain 0
            'target_domain': target_domain
        }
    
    def __len__(self):
        return len(self.source_files)
    
    def get_fixed_samples(self, num_samples=4):
        """Get fixed samples for visualization during training."""
        samples = {
            'source': [],
            'targets': {}
        }
        
        # Get fixed source samples
        for i in range(min(num_samples, len(self.source_files))):
            source_img = Image.open(self.source_files[i]).convert('RGB')
            source_img = self.transform(source_img)
            samples['source'].append(source_img)
        
        # Get fixed target samples from each domain
        for domain_name, files in self.target_files_by_domain.items():
            domain_idx = self.domain_to_idx[domain_name]
            samples['targets'][domain_idx] = []
            for i in range(min(num_samples, len(files))):
                target_img = Image.open(files[i]).convert('RGB')
                target_img = self.transform(target_img)
                samples['targets'][domain_idx].append(target_img)
        
        return samples