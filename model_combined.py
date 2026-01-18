
import torch
import torch.nn as nn
from model_2d_cnn import CNN2D
from model_1d_cnn import CNN1D

class CombinedModel(nn.Module):
    def __init__(self, input_dim_1d=89):
        super(CombinedModel, self).__init__()
        
        self.model_2d = CNN2D()
        self.model_1d = CNN1D(input_dim_1d)
        
        # Remove the final prediction layers (last Linear(..., 1)) from both
        # 2D FC: ... -> Linear(512, 1). We want output of 512.
        # But the FC in CNN2D is a Sequential. We can reconstruct or slice it.
        # CNN2D.fc has: Linear, BN, ReLU, Drop, Linear, BN, ReLU, Drop, Linear(1).
        # We want everything except the last Linear.
        # Let's retain the full Sequential but modify the end.
        self.model_2d.fc = nn.Sequential(*list(self.model_2d.fc.children())[:-1])
        # Output dim: 512
        
        # 1D FC: ... -> Linear(256, 1).
        self.model_1d.fc = nn.Sequential(*list(self.model_1d.fc.children())[:-1])
        # Output dim: 256
        
        # 2.3 Combined Model
        # Merge features (512 + 256 = 768)
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, img, metrics):
        # 2D path
        x2 = self.model_2d.layer1(img)
        x2 = self.model_2d.block1(x2)
        x2 = self.model_2d.block2(x2)
        x2 = self.model_2d.block3(x2)
        x2 = x2.view(x2.size(0), -1)
        feat_2d = self.model_2d.fc(x2)
        
        # 1D path
        x1 = metrics.unsqueeze(1)
        x1 = self.model_1d.layer1(x1)
        x1 = self.model_1d.layer2(x1)
        x1 = self.model_1d.layer3(x1)
        x1 = self.model_1d.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        feat_1d = self.model_1d.fc(x1)
        
        # Merge
        combined = torch.cat((feat_2d, feat_1d), dim=1)
        out = self.final_fc(combined)
        return out
