import torch
from torch import nn
import torch.nn.functional as F

class DeCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest', last_activation=True):
        super(DeCNNLayer, self).__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or scale_factor != 1:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        if last_activation:
            self.last_activation = nn.GELU()
        else:
            self.last_activation = None

    def forward(self, x):
        residual = self.shortcut(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv_block(x)
        x += residual
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x 