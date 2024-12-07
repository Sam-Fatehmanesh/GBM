import torch
from torch import nn
import pdb

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNNLayer, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        res = x
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = x + res
        return x

# Deconvolutional layer
class DeCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DeCNNLayer, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.deconv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        res = x
        x = self.deconv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = x + res
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
