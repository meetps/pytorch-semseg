import torch.nn as nn

from utils import *

class unet(nn.Module):

    def __init__(self, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        self.down1 = unetDown(self.in_channels, 64, self.is_batchnorm)
        self.down2 = unetDown(64, 128, self.is_batchnorm)
        self.down3 = unetDown(128, 256, self.is_batchnorm)
        self.down4 = unetDown(256, 512, self.is_batchnorm)
        self.center = unetConv2(512, 1024, self.is_batchnorm)
        self.up4 = unetUp(1024, 512, self.is_deconv)
        self.up3 = unetUp(512, 256, self.is_deconv)
        self.up2 = unetUp(256, 128, self.is_deconv)
        self.up1 = unetUp(128, 64, self.is_deconv)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)

        return self.final(up1)
