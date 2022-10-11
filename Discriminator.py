import torch 
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2))
        self.mlp2 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2))
        self.mlp3 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.2))
        self.mlp4 = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.2))
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.out(x)
        return x