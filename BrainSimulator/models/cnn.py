import torch
from torch import nn
import torch.nn.functional as F

import pdb

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNNLayer, self).__init__()
        
        # Calculate same padding
        padding = (kernel_size - 1) // 2  # This ensures output size matches input size
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        # Add debug print statements
        if out.shape != residual.shape:
            print(f"Shape mismatch - conv_block output: {out.shape}, residual: {residual.shape}")
        return out + residual

class DeCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest', kernel_size=3, last_activation=True):
        super(DeCNNLayer, self).__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode

        # Calculate same padding
        padding = (kernel_size - 1) // 2  # This ensures output size matches input size
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
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
        
        if self.last_activation is not None:
            x = self.last_activation(x)
            x += residual
            return x

        x += residual
        return x

# Basically the nn.F.interpolate func in class form
class InterpolateLayer(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super(InterpolateLayer, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
