
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from feature_extraction import extract_global_features

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, mode='2d'):
        """
        mode: '2d' (returns images), '1d' (returns metrics + global features), 'both'
        """
        self.num_samples = num_samples
        self.mode = mode
        
        # Paper says 74 metrics + 15 global features = 89 features total for 1D
        self.num_metrics = 74
        self.num_global_features = 15
        
        # Generate dummy data
        print(f"Generating {num_samples} dummy samples...")
        self.images = []
        self.metrics = []
        self.labels = []
        
        for _ in range(num_samples):
            # 144x144x3 random image (0-255)
            img = np.random.randint(0, 256, (144, 144, 3), dtype=np.uint8)
            self.images.append(img)
            
            # Random metrics (0-1000)
            met_vec = np.random.rand(self.num_metrics) * 1000
            self.metrics.append(met_vec)
            
            # Random CTR label (0-1)
            self.labels.append(np.random.rand())
            
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # Precompute global features for 1D mode
        if mode in ['1d', 'both']:
            print("Extracting global features...")
            self.combined_features = []
            for i in range(num_samples):
                g_feats = extract_global_features(self.images[i])
                
                # Normalize metrics (Min-Max) - simplified per sample for dummy
                # Real implementation would compute global min/max
                m_vec = self.metrics[i]
                m_norm = (m_vec - m_vec.min()) / (m_vec.max() - m_vec.min() + 1e-6)
                
                # Combine normalized metrics + global features
                combined = np.concatenate([m_norm, g_feats])
                self.combined_features.append(combined.astype(np.float32))
                
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.mode == '2d':
            # PyTorch expects Channels First: (C, H, W)
            # Image is (H, W, C), swap axes
            img = self.images[idx]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return img_tensor, label
            
        elif self.mode == '1d':
            feat = torch.from_numpy(self.combined_features[idx])
            return feat, label
            
        elif self.mode == 'both':
            img = self.images[idx]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            feat = torch.from_numpy(self.combined_features[idx])
            return (img_tensor, feat), label

def get_dataloader(num_samples=100, batch_size=32, mode='2d'):
    dataset = DummyDataset(num_samples, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
