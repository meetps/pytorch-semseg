import torch.nn as nn

from .utils3d import *


class unet3d(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        #filters = [64, 128, 256, 512, 1024]   #level4
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3d(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        #self.conv4 = unetConv3d(filters[2], filters[3], self.is_batchnorm)   #level4
        #self.maxpool4 = nn.MaxPool3d(kernel_size=2)                          #level4

        #self.center = unetConv3d(filters[3], filters[4], self.is_batchnorm)  #level4
        self.center = unetConv3d(filters[2], filters[3], self.is_batchnorm)

        # upsampling
        #self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)   #level4
        self.up_concat3 = unetUp3d(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        network_log('=>UNET3D=>inputs:{} '.format(inputs.size()), color_idx=None)

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        network_log('=>UNET3D=>conv1:{} and maxpoo1:{}'.format(conv1.size(), maxpool1.size()), color_idx=None)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        network_log('=>UNET3D=>conv2:{} and maxpoo2:{}'.format(conv2.size(), maxpool2.size()), color_idx=None)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        network_log('=>UNET3D=>conv3:{} and maxpoo3:{}'.format(conv3.size(), maxpool3.size()), color_idx=None)

        #conv4 = self.conv4(maxpool3)                                                           #level4
        #maxpool4 = self.maxpool4(conv4)                                                        #level4
        #network_log('=>UNET3D=>conv4:{} and maxpoo4:{}'.format(conv4.size(), maxpool4.size()), color_idx=None)  #level4

        #center = self.center(maxpool4)                                                         #level4
        center = self.center(maxpool3)
        network_log('=>UNET3D=>center:{}'.format(center.size()), color_idx=None)

        #up4 = self.up_concat4(conv4, center)                                                   #level4
        #network_log('=>UNET3D=>up4:{}'.format(up4.size()))                                             #level4
        #up3 = self.up_concat3(conv3, up4)                                                      #level4
        up3 = self.up_concat3(conv3, center)
        network_log('=>UNET3D=>up3:{}'.format(up3.size()), color_idx=None)
        up2 = self.up_concat2(conv2, up3)
        network_log('=>UNET3D=>up2:{}'.format(up2.size()), color_idx=None)
        up1 = self.up_concat1(conv1, up2)
        network_log('=>UNET3D=>up1:{}'.format(up1.size()), color_idx=None)

        final = self.final(up1)
        network_log('=>UNET3D=>final:{}'.format(final.size()))

        return final
