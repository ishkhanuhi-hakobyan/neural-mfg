import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_features, activation):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        self.activation = activation
        self.skip_connection_weight = 0.5

    def forward(self, x):
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out = self.skip_connection_weight * identity + out
        return out
