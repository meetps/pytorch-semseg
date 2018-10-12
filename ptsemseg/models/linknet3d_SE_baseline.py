
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


class linknet3d_SE_baseline(nn.Module):

    def __init__(self, resnet='resnet18', feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(linknet3d_SE_baseline, self).__init__()
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

        # Squeeze and Excitation
        self.linear_e1_1 = nn.Linear(64, 4)
        self.linear_e1_2 = nn.Linear(4, 64)
        self.linear_e2_1 = nn.Linear(128, 8)
        self.linear_e2_2 = nn.Linear(8, 128)
        self.linear_e3_1 = nn.Linear(256, 16)
        self.linear_e3_2 = nn.Linear(16, 256)
        self.linear_e4_1 = nn.Linear(512, 32)
        self.linear_e4_2 = nn.Linear(32, 512)

        self.linear_d4_1 = nn.Linear(256, 16)
        self.linear_d4_2 = nn.Linear(16, 256)
        self.linear_d3_1 = nn.Linear(128, 8)
        self.linear_d3_2 = nn.Linear(8, 128)
        self.linear_d2_1 = nn.Linear(64, 4)
        self.linear_d2_2 = nn.Linear(4, 64)
        self.linear_d1_1 = nn.Linear(64, 4)
        self.linear_d1_2 = nn.Linear(4, 64)


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
        network_log('linknet3d=>input.size():{}'.format(input.size()), color_idx=1)
        input1 = self.convbnrelu1(input)
        network_log('linknet3d=>input1.size():{}'.format(input1.size()), color_idx=1)
        input2 = self.maxpool(input1)
        network_log('linknet3d=>input2.size():{}'.format(input2.size()), color_idx=1)

        e1 = self.encoder1(input2)
        network_log('linknet3d=>e1.size():{}'.format(e1.size()), color_idx=2)
        e1_GlobalAvgPooling = F.avg_pool3d(e1, kernel_size=e1.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>e1_GlobalAvgPooling.size():{}'.format(e1_GlobalAvgPooling.size()), color_idx=3)
        e1_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_e1_2(F.relu(self.linear_e1_1(e1_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>e1_FC.size():{}'.format(e1_FC.size()), color_idx=3)
        e1_rescale = e1 * e1_FC.expand_as(e1)
        network_log('linknet3d=>e1_rescale.size():{}'.format(e1_rescale.size()), color_idx=3)


        e2 = self.encoder2(e1_rescale)
        network_log('linknet3d=>e2.size():{}'.format(e2.size()), color_idx=2)
        e2_GlobalAvgPooling = F.avg_pool3d(e2, kernel_size=e2.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>e2_GlobalAvgPooling.size():{}'.format(e2_GlobalAvgPooling.size()), color_idx=3)
        e2_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_e2_2(F.relu(self.linear_e2_1(e2_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>e2_FC.size():{}'.format(e2_FC.size()), color_idx=3)
        e2_rescale = e2 * e2_FC.expand_as(e2)
        network_log('linknet3d=>e2_rescale.size():{}'.format(e2_rescale.size()), color_idx=3)

        e3 = self.encoder3(e2_rescale)
        network_log('linknet3d=>e3.size():{}'.format(e3.size()), color_idx=2)
        e3_GlobalAvgPooling = F.avg_pool3d(e3, kernel_size=e3.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>e3_GlobalAvgPooling.size():{}'.format(e3_GlobalAvgPooling.size()), color_idx=3)
        e3_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_e3_2(F.relu(self.linear_e3_1(e3_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>e3_FC.size():{}'.format(e3_FC.size()), color_idx=3)
        e3_rescale = e3 * e3_FC.expand_as(e3)
        network_log('linknet3d=>e3_rescale.size():{}'.format(e3_rescale.size()), color_idx=3)

        e4 = self.encoder4(e3_rescale)
        network_log('linknet3d=>e4.size():{}'.format(e4.size()), color_idx=2)
        e4_GlobalAvgPooling = F.avg_pool3d(e4, kernel_size=e4.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>e4_GlobalAvgPooling.size():{}'.format(e4_GlobalAvgPooling.size()), color_idx=3)
        e4_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_e4_2(F.relu(self.linear_e4_1(e4_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>e4_FC.size():{}'.format(e4_FC.size()), color_idx=3)
        e4_rescale = e4 * e4_FC.expand_as(e4)
        network_log('linknet3d=>e4_rescale.size():{}'.format(e4_rescale.size()), color_idx=3)

        d4 = self.decoder4(e4_rescale) + e3
        network_log('linknet3d=>d4.size():{}'.format(d4.size()), color_idx=1)
        d4_GlobalAvgPooling = F.avg_pool3d(d4, kernel_size=d4.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>d4_GlobalAvgPooling.size():{}'.format(d4_GlobalAvgPooling.size()), color_idx=3)
        d4_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_d4_2(F.relu(self.linear_d4_1(d4_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>d4_FC.size():{}'.format(d4_FC.size()), color_idx=3)
        d4_rescale = d4 * d4_FC.expand_as(d4)
        network_log('linknet3d=>d4_rescale.size():{}'.format(d4_rescale.size()), color_idx=3)

        d3 = self.decoder3(d4_rescale) + e2
        network_log('linknet3d=>d3.size():{}'.format(d3.size()), color_idx=1)
        d3_GlobalAvgPooling = F.avg_pool3d(d3, kernel_size=d3.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>d3_GlobalAvgPooling.size():{}'.format(d3_GlobalAvgPooling.size()), color_idx=3)
        d3_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_d3_2(F.relu(self.linear_d3_1(d3_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>d3_FC.size():{}'.format(d3_FC.size()), color_idx=3)
        d3_rescale = d3 * d3_FC.expand_as(d3)
        network_log('linknet3d=>d3_rescale.size():{}'.format(d3_rescale.size()), color_idx=3)

        d2 = self.decoder2(d3_rescale) + e1
        network_log('linknet3d=>d3.size():{}'.format(d2.size()), color_idx=1)
        d2_GlobalAvgPooling = F.avg_pool3d(d2, kernel_size=d2.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>d2_GlobalAvgPooling.size():{}'.format(d2_GlobalAvgPooling.size()), color_idx=3)
        d2_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_d2_2(F.relu(self.linear_d2_1(d2_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>d2_FC.size():{}'.format(d2_FC.size()), color_idx=3)
        d2_rescale = d2 * d2_FC.expand_as(d2)
        network_log('linknet3d=>d2_rescale.size():{}'.format(d2_rescale.size()), color_idx=3)

        d1 = self.decoder1(d2_rescale)
        network_log('linknet3d=>d1.size():{}'.format(d1.size()), color_idx=1)
        d1_GlobalAvgPooling = F.avg_pool3d(d1, kernel_size=d1.size()[2:]).squeeze(2).squeeze(2).squeeze(2)
        network_log('linknet3d=>d1_GlobalAvgPooling.size():{}'.format(d1_GlobalAvgPooling.size()), color_idx=3)
        d1_FC = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(F.sigmoid(self.linear_d1_2(F.relu(self.linear_d1_1(d1_GlobalAvgPooling), inplace=True))), 2), 3), 4)
        network_log('linknet3d=>d1_FC.size():{}'.format(d1_FC.size()), color_idx=3)
        d1_rescale = d1 * d1_FC.expand_as(d1)
        network_log('linknet3d=>d1_rescale.size():{}'.format(d1_rescale.size()), color_idx=3)

        f1 = self.finaldeconvbnrelu1(d1_rescale)
        network_log('linknet3d=>f1.size():{}'.format(f1.size()), color_idx=2)
        f2 = self.finalconvbnrelu2(f1)
        network_log('linknet3d=>f2.size():{}'.format(f2.size()), color_idx=2)
        f3 = self.finalconv3(f2)
        network_log('linknet3d=>f3.size():{}'.format(f3.size()), color_idx=2)
        return f3