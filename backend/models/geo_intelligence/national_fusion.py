import numpy as np
import torch
from torch import nn


class NationalFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('band_mean', torch.zeros(1, 9, 1, 1))
        self.register_buffer('band_std', torch.ones(1, 9, 1, 1))
        self.register_buffer('gravity_mean', torch.zeros(1))
        self.register_buffer('gravity_std', torch.ones(1))
        self.encoder = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(), nn.AvgPool2d(2),
                                     nn.Conv2d(16, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(), nn.AvgPool2d(2),
                                     nn.Conv2d(24, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.gravity = nn.Sequential(nn.Linear(1, 8), nn.SiLU())
        self.head = nn.Sequential(nn.Dropout(.4), nn.Linear(40, 16), nn.SiLU(), nn.Dropout(.3), nn.Linear(16, 1))

    def fit_normalization(self, images, gravity):
        self.band_mean.copy_(images.mean((0, 2, 3), keepdim=True))
        self.band_std.copy_(images.std((0, 2, 3), keepdim=True).clamp_min(.02))
        self.gravity_mean.copy_(gravity.mean().reshape(1))
        self.gravity_std.copy_(gravity.std().clamp_min(1).reshape(1))

    def forward(self, images, gravity):
        image_features = self.encoder((images-self.band_mean)/self.band_std).flatten(1)
        g = self.gravity((gravity-self.gravity_mean)/self.gravity_std)
        return self.head(torch.cat([image_features, g], dim=1)).flatten()


class Ensemble(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.models = nn.ModuleList([NationalFusionCNN() for _ in states])
        for model, state in zip(self.models, states):
            model.load_state_dict(state)

    def forward(self, images, gravity):
        probabilities = torch.stack([torch.sigmoid(m(images, gravity)) for m in self.models]).mean(0)
        return torch.logit(probabilities.clamp(1e-6, 1-1e-6))


def augment(images, generator):
    a = images.clone()
    if generator.random() < .5:
        a = a.flip(-1)
    if generator.random() < .5:
        a = a.flip(-2)
    a = a.rot90(int(generator.integers(4)), (-2, -1))
    gain = torch.as_tensor(generator.uniform(.95, 1.05, (len(a), 1, 1, 1)), dtype=a.dtype)
    a[:, :6] = (a[:, :6]*gain).clamp(1e-5, 1.5)
    for channel, (i, j) in enumerate([(3, 2), (1, 3), (3, 4)], 6):
        a[:, channel] = (a[:, i]-a[:, j])/(a[:, i]+a[:, j]).clamp_min(1e-8)
    return a
