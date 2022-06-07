import torch
from torch import nn




class WSConv2d(nn.Module):
    """
    instead of scale the weight at running time, we scale the input feature map 
    """
    def __init__(self, in_channel, out_channel, k_s=3, s=1, p=1, gain=2) -> None:
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, k_s, s, p)

        self.scale = (gain / in_channel * (k_s ** 2)) ** 0.5

        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

        # only scale the weights, keep the bias
        self.bias = self.conv.bias.view(-1, out_channel, 1, 1)
        self.conv.bias = None

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias
        

class PixelNorm(nn.Module):
    """
    norm input tensor to unit length in each pixel at each feature map"""
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class UpSampling(nn.Module):
    """
    upsampling input tensor by nearest neibou"""
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class ConvBlock(nn.Module):
    """
    structure: WSConv - LeakyReLU - PixelNorm"""
    def __init__(self, in_channel, out_channel) -> None:
        super(ConvBlock, self).__init__()
        block = []
        block.append(WSConv2d(in_channel, out_channel))
        block.append(nn.LeakyReLU(0.2))
        block.append(PixelNorm())
        block.append(WSConv2d(out_channel, out_channel))
        block.append(nn.LeakyReLU(0.2))
        block.append(PixelNorm())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


