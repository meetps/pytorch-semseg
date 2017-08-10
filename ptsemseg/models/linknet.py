import torch.nn as nn

from utils import *

Resnets = {'resnet18' :{'layers':[2, 2, 2, 2],'filters':[64, 128, 256, 512], 'block':residualBlock,'expansion':1},
           'resnet34' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBlock,'expansion':1},
           'resnet50' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4},
           'resnet101' :{'layers':[3, 4, 23, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4},
           'resnet152':{'layers':[3, 8, 36, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4}
            }


class linknet(nn.Module):

    def __init__(self, resnet='resnet18', feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(pspnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        assert resnet in Resnets.keys(), 'Not a valid resnet, currently supported resnets are 18, 34, 50, 101 and 152'
        layers = Resnets[resnet]['layers']
        filters = Resnets[resnet]['filters']
        # filters = [x / self.feature_scale for x in filters]
        expansion =Resnets[resnet]['expansion']

        self.inplanes = filters[0]


        # Encoder
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels=3, k_size=7, n_filters=64,
                                               padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Resnets[resnet]['block']
        self.encoder1 = self._make_layer(block, filters[0], layers[0])
        self.encoder2 = self._make_layer(block, filters[1], layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)


        # Decoder
        self.decoder4 = linknetUp(filters[3]*expansion, filters[2]*expansion)
        self.decoder3 = linknetUp(filters[2]*expansion, filters[1]*expansion)
        self.decoder2 = linknetUp(filters[1]*expansion, filters[0]*expansion)
        self.decoder1 = linknetUp(filters[0]*expansion, filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = deconv2DBatchNormRelu(filters[0], 32/feature_scale, 2, 2, 0)
        self.finalconvbnrelu2 = conv2DBatchNormRelu(in_channels=32/feature_scale, k_size=3, n_filters=32/feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv2d(int(32/feature_scale), int(n_classes), 3, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv2DBatchNorm(self.inplanes, planes*block.expansion, k_size=1, stride=stride, padding=0, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.convbnrelu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4)
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        d2 = d2 + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)

        return f3