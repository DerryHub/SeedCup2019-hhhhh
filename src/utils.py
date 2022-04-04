from torch import nn
import torch


class MyCost(nn.Module):
    def __init__(self, config):
        super(MyCost, self).__init__()
        self.config = config

    def forward(self, x, y):
        delta = x - y
         #scale = torch.sigmoid(delta) * self.config.scale
        scale = (torch.sign(delta)+1)*0.5*self.config.scale
        return torch.sum(torch.pow(delta, 2) * (scale + 1))

