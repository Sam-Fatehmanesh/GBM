import torch
from torch import nn
import pdb

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNNLayer, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # Add a projection shortcut if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block(x)
        x = x + residual
        return x

# Deconvolutional layer
class DeCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DeCNNLayer, self).__init__()
        
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # # Add a projection shortcut if dimensions change
        # self.shortcut = nn.Identity()
        # if in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride),
        #         nn.BatchNorm2d(out_channels)
        #     )
        

    def forward(self, x):
        #residual = self.shortcut(x)
        x = self.deconv_block(x)
        #x = x + residual
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
