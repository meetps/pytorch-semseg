DEBUG=True
colors = ['\x1b[0m', '\x1b[6;30;42m', '\x1b[2;30;41m']
def log(s, color):
    if DEBUG:
        print(color+s+colors[0])
###################
import torch.nn as nn
import torchvision.models as models

from .utils3d import *

Resnets = {'resnet18' :{'layers':[2, 2, 2, 2],'filters':[64, 128, 256, 512], 'block':residualBlock3D,'expansion':1},
           'resnet34' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBlock3D,'expansion':1},
           'resnet50' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBlock3D,'expansion':4},
           'resnet101' :{'layers':[3, 4, 23, 3],'filters':[64, 128, 256, 512], 'block':residualBlock3D,'expansion':4},
           'resnet152':{'layers':[3, 8, 36, 3],'filters':[64, 128, 256, 512], 'block':residualBlock3D,'expansion':4}
            }


class linknet3d(nn.Module):

    def __init__(self, resnet='resnet18', feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(linknet3d, self).__init__()
        self.n_classes=n_classes
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
        self.convbnrelu1 = conv3DBatchNormRelu(in_channels=3, k_size=7, n_filters=64,
                                               padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        block = Resnets[resnet]['block']

        self.encoder1 = self._make_layer(block, filters[0], layers[0])
        self.encoder2 = self._make_layer(block, filters[1], layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], layers[3], stride=2)

        self.avgpool = nn.AvgPool3d(7)


        # Decoder
        self.decoder4 = linknetUp3D(filters[3] * expansion, filters[2] * expansion)
        self.decoder3 = linknetUp3D(filters[2] * expansion, filters[1] * expansion)
        self.decoder2 = linknetUp3D(filters[1] * expansion, filters[0] * expansion)
        self.decoder1 = linknetUp3D(filters[0] * expansion, filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = deconv3DBatchNormRelu(filters[0], 32/feature_scale, 2, 2, 0)
        self.finalconvbnrelu2 = conv3DBatchNormRelu(in_channels=32/feature_scale, k_size=3, n_filters=32/feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv3d(int(32/feature_scale), 2, 3, 1, 1)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv3DBatchNorm(self.inplanes, planes*block.expansion, k_size=1, stride=stride, padding=0, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        # Encoder
        log('linknet3d=>input.size():{}'.format(input.size()), colors[1])
        input1 = self.convbnrelu1(input)
        log('linknet3d=>input1.size():{}'.format(input1.size()), colors[1])
        input2 = self.maxpool(input1)
        log('linknet3d=>input2.size():{}'.format(input2.size()), colors[1])
        e1 = self.encoder1(input2)
        log('linknet3d=>e1.size():{}'.format(e1.size()), colors[2])
        e2 = self.encoder2(e1)
        log('linknet3d=>e2.size():{}'.format(e2.size()), colors[2])
        e3 = self.encoder3(e2)
        log('linknet3d=>e3.size():{}'.format(e3.size()), colors[2])
        e4 = self.encoder4(e3)
        log('linknet3d=>e4.size():{}'.format(e4.size()), colors[2])

        d4 = self.decoder4(e4) + e3
        log('linknet3d=>d4.size():{}'.format(d4.size()), colors[1])
        d3 = self.decoder3(d4) + e2
        log('linknet3d=>d3.size():{}'.format(d3.size()), colors[1])
        d2 = self.decoder2(d3) + e1
        log('linknet3d=>d3.size():{}'.format(d2.size()), colors[1])
        d1 = self.decoder1(d2)
        log('linknet3d=>d1.size():{}'.format(d1.size()), colors[1])
        f1 = self.finaldeconvbnrelu1(d1)
        log('linknet3d=>f1.size():{}'.format(f1.size()), colors[2])
        f2 = self.finalconvbnrelu2(f1)
        log('linknet3d=>f2.size():{}'.format(f2.size()), colors[2])
        f3 = self.finalconv3(f2)
        log('linknet3d=>f3.size():{}'.format(f3.size()), colors[2])
        return f3