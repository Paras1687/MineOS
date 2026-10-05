import torch
from torch import nn

class ProspectivityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(9,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1),nn.GroupNorm(4,48),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1),nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(.25),nn.Linear(64,1))
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1)).squeeze(1)
