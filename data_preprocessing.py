
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
from feature_extraction import extract_global_features

class AdDataset(Dataset):
    def __init__(self, images, metrics, labels, transform=None, extract_features=True):
        """
        images: numpy array (N, H, W, 3)
        metrics: numpy array (N, 74)
        labels: numpy array (N, 1)
        """
        self.images = images
        self.metrics = metrics
        self.labels = labels
        self.transform = transform
        self.extract_features = extract_features
        
        # Normalize metrics (Min-Max)
        # In a real scenario, fits on train, applies to test. 
        # Here we do global for simplicity of the dummy data block.
        min_vals = np.min(self.metrics, axis=0)
        max_vals = np.max(self.metrics, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1 # Avoid div by zero
        self.metrics_norm = (self.metrics - min_vals) / range_vals

        # Precompute global features if needed (or compute on fly)
        # Computed on fly allows augmentation to affect them? 
        # Paper implies they are static properties of the image? 
        # "Global features ... from display images". 
        # Calculating on fly is slower but correct if we want aug to affect them.
        # However, usually global features are robust. Let's precompute for speed 
        # BUT if we augment (color jitter), they change.
        # User plan says "1.1 Image Data Processing... Implement data augmentation".
        # We will compute on the fly to reflect augmentations if possible, 
        # OR precompute on original images if aug is just for CNN robustness.
        # Let's precompute on original to save time as per typical tabular+image pipelines.
        if self.extract_features:
            print("Extracting global features for all images...")
            self.global_features = []
            for img in self.images:
                self.global_features.append(extract_global_features(img))
            self.global_features = np.array(self.global_features, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image
        img = self.images[idx] # H, W, 3
        
        # Convert to PIL/Tensor for transforms
        # ToTensor converts (H,W,C) [0,255] -> (C,H,W) [0.0, 1.0]
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        # Metrics
        metric_vec = self.metrics_norm[idx]
        
        # Global Features
        if self.extract_features:
            glob_feat = self.global_features[idx]
            # Combined 1D input: Normalized Metrics + Global Features
            combined_1d = np.concatenate([metric_vec, glob_feat]).astype(np.float32)
        else:
            combined_1d = metric_vec.astype(np.float32)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Return tuple: (image_input, combined_1d_input), label
        return (img_tensor, torch.from_numpy(combined_1d)), label


def get_data_loaders(images, metrics, labels, batch_size=32):
    # 1.1 Image Data Processing
    # Resize to 144x144 (Assumed images are already 144 or resized before)
    # Augmentation: Rotation, Flip, Zoom (approximated by RandomResizedCrop or Affine)
    
    train_transform = transforms.Compose([
        transforms.Resize((144, 144)), # Ensure size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # Zoom: RandomResizedCrop can simulate zoom
        transforms.RandomResizedCrop(144, scale=(0.8, 1.0)), 
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((144, 144))
    ])
    
    # 1.3 splitting
    dataset_size = len(images)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    
    # We need to split the raw arrays first to avoid leakage in normalization 
    # (Ideally), but `AdDataset` does norm internally for simplicity here. 
    # Let's create one dataset and split, noting standard transforms are applied.
    # Note: Applying different transforms to train/test with `random_split` is tricky 
    # because they share the underlying dataset object.
    # Approach: Wrap separate subsets with transforms.
    
    full_dataset = AdDataset(images, metrics, labels, transform=None, extract_features=True)
    
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])
    
    # Apply transforms via a wrapper or modifying the dataset for the subset?
    # Cleaner: Split raw data then create two datasets.
    
    # Split indices
    indices = np.random.permutation(dataset_size)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    train_dataset = AdDataset(
        images[train_idx], metrics[train_idx], labels[train_idx], 
        transform=train_transform, extract_features=True
    )
    
    test_dataset = AdDataset(
        images[test_idx], metrics[test_idx], labels[test_idx], 
        transform=test_transform, extract_features=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
