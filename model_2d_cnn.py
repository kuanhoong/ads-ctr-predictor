
import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        
        # 2.1: 13 convolutional layers
        # Layer 1: Conv(5x5, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Blocks: [Conv(3x3, 32) x4 + MaxPool] x3 ? 
        # Paper says: "three blocks of convolution operations... each contains 4 conv layers..."
        # User prompt check: "Architecture: Conv(5x5, 32) → [Conv(3x3, 32) x4 + MaxPool] x3"
        # My previous model increased channels (32->64->128). 
        # User plan explicitly says "Conv(3x3, 32) x4". It implies keeping 32 filters constant?
        # "Conv(5x5, 32) -> [Conv(3x3, 32) x4 + MaxPool] x3"
        # Reading literally: All blocks use 32 filters. 
        # Usually channels increase as spatial dims decrease, but I will follow the user's explicit notation "32".
        
        def make_block(in_channels, out_channels):
            layers = []
            c_in = in_channels
            for _ in range(4):
                layers.extend([
                    nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ])
                c_in = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # Block 1, 2, 3
        self.block1 = make_block(32, 32)
        # Input to block 2 is 32, output 32
        self.block2 = make_block(32, 32)
        # Input to block 3 is 32, output 32 
        self.block3 = make_block(32, 32)
        
        # Output size calculation:
        # Input: 144x144
        # Layer 1: 144x144
        # Block 1 Pool: 72x72
        # Block 2 Pool: 36x36
        # Block 3 Pool: 18x18
        # Final Flatten: 32 * 18 * 18 = 10368
        
        # FC Layers: 512x512
        self.fc = nn.Sequential(
            nn.Linear(32 * 18 * 18, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Standard to add dropout in robust FC
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
