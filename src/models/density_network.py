import torch
import torch.nn as nn

from src.models.resnet import ResNetBlock


class DensityNetwork(nn.Module):
    def __init__(self):
        super(DensityNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 100)  # Input now includes both t and x
        self.res_block1 = ResNetBlock(100, nn.Softplus())
        self.res_block2 = ResNetBlock(100, nn.Softplus())
        self.res_block3 = ResNetBlock(100, nn.Softplus())
        self.fc2 = nn.Linear(100, 1)

    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)  # Concatenate t and x along the feature dimension
        x = self.fc1(tx)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.fc2(x)
        return x
