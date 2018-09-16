
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


class linknet3d_exp(nn.Module):

    def __init__(self, resnet='resnet18', feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True, n_macroblocks=None):
        super(linknet3d_exp, self).__init__()
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

        # Decoder
        self.decoder4 = linknetUp3D(filters[3] * expansion, filters[2])
        self.decoder3 = linknetUp3D(filters[2] * expansion, filters[1])
        self.decoder2 = linknetUp3D(filters[1] * expansion, filters[0])
        self.decoder1 = linknetUp3D(filters[0] * expansion, filters[0])


        # macroblock classification
        self.n_macroblocks = n_macroblocks
        self.linear = nn.Linear(filters[3], self.n_macroblocks)

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
    def fusion(selfs, f1, f2):
        return f1 + f2
        #return torch.cat([f1, f2], 1)
    def forward(self, input):
        # Encoder
        network_log('linknet3d=>input.size():{}'.format(input.size()), color_idx=1)
        input1 = self.convbnrelu1(input)
        network_log('linknet3d=>input1.size():{}'.format(input1.size()), color_idx=1)
        input2 = self.maxpool(input1)
        network_log('linknet3d=>input2.size():{}'.format(input2.size()), color_idx=1)
        e1 = self.encoder1(input2)
        network_log('linknet3d=>e1.size():{}'.format(e1.size()), color_idx=2)
        e2 = self.encoder2(e1)
        network_log('linknet3d=>e2.size():{}'.format(e2.size()), color_idx=2)
        e3 = self.encoder3(e2)
        network_log('linknet3d=>e3.size():{}'.format(e3.size()), color_idx=2)
        e4 = self.encoder4(e3)
        network_log('linknet3d=>e4.size():{}'.format(e4.size()), color_idx=2)

        #d4 = self.decoder4(e4) + e3
        d4 = self.decoder4(e4)
        network_log('linknet3d=>d4.size():{}'.format(d4.size()), color_idx=1)
        d4_fusion = self.fusion(d4, e3)
        network_log('linknet3d=>d4_cat.size():{}'.format(d4_fusion.size()), color_idx=1)

        d3 = self.decoder3(d4_fusion)
        network_log('linknet3d=>d3.size():{}'.format(d3.size()), color_idx=1)
        d3_fusion = self.fusion(d3, e2)
        network_log('linknet3d=>d3_cat.size():{}'.format(d3_fusion.size()), color_idx=1)

        d2 = self.decoder2(d3_fusion)
        network_log('linknet3d=>d2.size():{}'.format(d2.size()), color_idx=1)
        d2_fusion = self.fusion(d2, e1)
        network_log('linknet3d=>d2_cat.size():{}'.format(d2_fusion.size()), color_idx=1)

        d1 = self.decoder1(d2_fusion)
        network_log('linknet3d=>d1.size():{}'.format(d1.size()), color_idx=1)
        f1 = self.finaldeconvbnrelu1(d1)
        network_log('linknet3d=>f1.size():{}'.format(f1.size()), color_idx=2)
        f2 = self.finalconvbnrelu2(f1)
        network_log('linknet3d=>f2.size():{}'.format(f2.size()), color_idx=2)
        f3 = self.finalconv3(f2)
        network_log('linknet3d=>f3.size():{}'.format(f3.size()), color_idx=2)

        mb4 = F.max_pool3d(e4, kernel_size=e4.size()[2:])
        mb4_flatten = mb4.view(-1, mb4.size()[1])
        mb = self.linear(mb4_flatten)
        network_log('linknet3d=>mb.size():{}'.format(mb.size()), color_idx=1)
        return f3, mb