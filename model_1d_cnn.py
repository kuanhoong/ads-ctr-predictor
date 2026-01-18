
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim, kernel_size=5):
        super(CNN1D, self).__init__()
        
        # 2.2: 3 convolutional layers
        # Conv(5x5) varying kernels (32, 64, 64)
        # Note: "Conv(5x5)" for 1D is kernel_size=5.
        
        # Layer 1: 1 -> 32
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 2: 32 -> 64
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 3: 64 -> 64
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Dimensionality after pools?
        # Input N (74+15=89).
        # L1: N/2 = 44
        # L2: N/4 = 22
        # L3: N/8 = 11
        # Flatten: 64 * 11 = 704 approx.
        # We can use AdaptiveAvgPool to fix size or just calculate dynamic
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # Valid strategy to handle variable/calc size easily
        # But if we want to follow FC(256x256), we need a specific input size? 
        # AdaptivePool1d(1) -> Output (Batch, 64, 1) -> Flatten -> 64.
        
        # FC Layer (256x256)
        # Input: 64 (from AdaptivePool)
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            # nn.Linear(input_flattened, 256) if not using adaptive
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # x: (Batch, InputDim) -> Unsqueeze (Batch, 1, InputDim)
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
