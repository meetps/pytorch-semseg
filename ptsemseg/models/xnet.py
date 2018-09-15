DEBUG=False
def log(s):
    if DEBUG:
        print(s)
###################
import torch.nn as nn

from ptsemseg.models.xnet_utils import *


class xnet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(xnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_batchnorm)
        self.avgpool1 = nn.AvgPool3d(kernel_size=2)

        self.conv2 = unetConv3d(filters[0], filters[1], self.is_batchnorm)
        self.avgpool2 = nn.AvgPool3d(kernel_size=2)

        self.center = unetConv3d(filters[1], filters[2], self.is_batchnorm)

        # upsampling
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        log('=>UNET3D=>inputs:{} '.format(inputs.size()))

        conv1 = self.conv1(inputs)
        avgpool1 = self.avgpool1(conv1)
        log('=>UNET3D=>conv1:{} and maxpoo1:{}'.format(conv1.size(), avgpool1.size()))

        conv2 = self.conv2(avgpool1)
        avgpool2 = self.avgpool2(conv2)
        log('=>UNET3D=>conv2:{} and maxpoo2:{}'.format(conv2.size(), avgpool2.size()))

        center = self.center(avgpool2)
        log('=>UNET3D=>center:{}'.format(center.size()))

        up2 = self.up_concat2(conv2, center)
        log('=>UNET3D=>up2:{}'.format(up2.size()))
        up1 = self.up_concat1(conv1, up2)
        log('=>UNET3D=>up1:{}'.format(up1.size()))

        final = self.final(up1)
        log('=>UNET3D=>final:{}'.format(final.size()))

        return final
