import torch.nn as nn

from utils import *

class segnet(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()
        
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown(self.in_channels,64)
        self.down2 = segnetDown(64, 128)
        self.down3 = segnetDown(128, 256)

        self.down4 = conv2DBatchNormRelu(256, 3, 512, 1)
        self.up4 = conv2DBatchNorm(512, 3, 512, 1)

        self.up3 = segnetUp(512, 256)
        self.up2 = segnetUp(256, 128)
        self.up1 = segnetUp(128, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, inputs):
        down1, indices_1 = self.down1(inputs)
        down2, indices_2 = self.down2(down1)
        down3, indices_3 = self.down3(down2)
        down4 = self.down4(down3)
        up4 = self.up4(down4)
        up3 = self.up3(up4, indices_3, down3.size())
        up2 = self.up2(up3, indices_2, down2.size())
        up1 = self.up1(up2, indices_1, down1.size())

        return self.final(up1)