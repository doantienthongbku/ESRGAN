import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseResidualBlock(nn.Module):
    """
        Residual Dense Block is the core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
        https://arxiv.org/abs/1802.08797
    """
    def __init__(self, channels, growth_channels, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        
        def _block(in_channels, non_linearity=True):
            layers = [nn.Conv2d(in_channels, growth_channels, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        
        self.b1 = _block(in_channels=channels + 0 * growth_channels)
        self.b2 = _block(in_channels=channels + 1 * growth_channels)
        self.b3 = _block(in_channels=channels + 2 * growth_channels)
        self.b4 = _block(in_channels=channels + 3 * growth_channels)
        self.b5 = _block(in_channels=channels + 4 * growth_channels, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], dim=1)
        return out.mul(self.res_scale) + x
    
    
class ResidualInResidualDenseBlock(nn.Module):
    """
        Core block of ESRGAN Generation
    """
    def __init__(self, in_channels, growth_channels, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(in_channels, growth_channels),
            DenseResidualBlock(in_channels, growth_channels),
            DenseResidualBlock(in_channels, growth_channels)
        )
        
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
        

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 channels=64, growth_channels=32, upscale_factor=4, 
                 res_scale=0.2, n_blocks=23, num_upsample=2):
        super(Generator, self).__init__()
        self.upscale_factor = upscale_factor
        # First Convolution
        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=True)
        
        # ResidualInResidualDense Blocks
        trunk = []
        for _ in range(n_blocks):
            trunk += [ResidualInResidualDenseBlock(channels, growth_channels, res_scale)]
        self.trunk = nn.Sequential(*trunk)
        
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        
        # Upsampling layers
        upsample_layer = []
        for _ in range(num_upsample):
            upsample_layer += [
                nn.Conv2d(channels, channels * (upscale_factor ** 2), 3, 1, 1, bias=True),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor)
            ]
        self.upsample_layer = nn.Sequential(*upsample_layer)
        
        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)
        )
        
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out