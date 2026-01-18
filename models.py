
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    """
    2D CNN for CTR prediction from Display Images.
    Paper Section 3.1:
    - Input: 144x144x3
    - Layer 1: 5x5 Conv
    - 3 Blocks of:
        - 4 Conv layers (3x3)
        - 1 MaxPool (2x2)
    - Total Conv layers = 1 + 3*4 = 13.
    """
    def __init__(self):
        super(CNN2D, self).__init__()
        
        # Layer 1: 5x5 Conv
        # Assuming standard filter count increase strategy (e.g., 32 -> 64 -> 128) 
        # Paper doesn't explicitly specify filter counts for every layer, just "kernels".
        # Let's infer reasonable defaults: 32 filters.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        
        # Block 1: 4 Conv (3x3), then MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 144 -> 72
        )
        
        # Block 2: 4 Conv (3x3), then MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 72 -> 36
        )
        
        # Block 3: 4 Conv (3x3), then MaxPool
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 36 -> 18
        )
        
        # Final prediction layer
        # Output is 18x18x128
        self.fc = nn.Linear(128 * 18 * 18, 1)

    def forward(self, x):
        # x: (N, 3, 144, 144)
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN1D(nn.Module):
    """
    1D CNN for CTR prediction from Click Metrics + Global Features.
    Paper Section 3.2:
    - Input: Vector x (size N)
    - Conv Layers with kernel K (e.g., 5).
    - 1D MaxPool.
    - 3 Convolution layers shown in Fig 3.
    """
    def __init__(self, input_dim, kernel_size=5):
        super(CNN1D, self).__init__()
        
        # Input is (Batch, 1, InputDim)
        # Layer 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate output size for FC
        # Approx reduction: Input -> /2 -> /4 -> /8
        # Let's do a dynamic check or AdaptiveAvgPool
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (N, InputDim) -> Unsqueeze to (N, 1, InputDim)
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
