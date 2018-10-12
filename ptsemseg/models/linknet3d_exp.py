
###################
import torch.nn as nn
import torchvision.models as models

from .utils3d import *

Resnets = {'resnet18' :{'layers':[2, 2, 2, 2],'filters':[64*4, 128*2, 256//2, 512//4], 'block':residualBlock3D_LOC,'expansion':1},  # pay attension that relut is missed
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
        self.convbnrelu1 = conv3DBatchNormRelu(in_channels=3, k_size=3, n_filters=filters[0],
                                               padding=1, stride=2, bias=False)
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
        self.relu = nn.ReLU(inplace=True)
        self.n_macroblocks = n_macroblocks
        #self.linear = nn.Linear(filters[0]+filters[0]+filters[1]+filters[2], self.n_macroblocks)
        self.linear = nn.Linear(filters[0]+filters[0], self.n_macroblocks)
        self.dropout = nn.Dropout(p=0.3)
        #self.linear = nn.Linear(filters[1]+filters[2]+filters[3], self.n_macroblocks)
        self.downsample1 = conv3DBatchNorm(filters[0], filters[0], k_size=1, stride=1, padding=0, bias=False)
        self.downsample2 = conv3DBatchNorm(filters[0], filters[0], k_size=1, stride=1, padding=0, bias=False)
        self.downsample2_ = conv3DBatchNorm(filters[0], filters[1], k_size=1, stride=2, padding=0, bias=False)
        self.downsample3 = conv3DBatchNorm(filters[1], filters[1], k_size=1, stride=1, padding=0, bias=False)
        self.downsample3_ = conv3DBatchNorm(filters[1], filters[2], k_size=1, stride=2, padding=0, bias=False)
        self.downsample4 = conv3DBatchNorm(filters[2], filters[2], k_size=1, stride=1, padding=0, bias=False)
        self.downsample4_ = conv3DBatchNorm(filters[2], filters[3], k_size=1, stride=2, padding=0, bias=False)


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
        network_log('[ConvFirst]\nlinknet3d=>input1.size():{}'.format(input1.size()), color_idx=1)
        input1_maxpool = self.maxpool(input1)
        network_log('linknet3d=>input1_maxpool.size():{}'.format(input1_maxpool.size()), color_idx=1)

        loc1_downsample = self.downsample1(input1_maxpool)
        network_log('[EB1]\nlinknet3d=>loc1_downsample.size():{}'.format(loc1_downsample.size()), color_idx=2)
        e1 = self.relu(self.encoder1(input1_maxpool + loc1_downsample))
        network_log('linknet3d=>e1.size():{}'.format(e1.size()), color_idx=2)

        loc2_downsample = self.downsample2(e1)
        network_log('[EB2]\nlinknet3d=>loc2_downsample.size():{}'.format(loc2_downsample.size()), color_idx=2)
        e2 = self.relu(self.encoder2(e1 + loc2_downsample))
        network_log('linknet3d=>e2.size():{}'.format(e2.size()), color_idx=2)

        loc3_downsample = self.downsample3(e2)
        network_log('[EB3]\nlinknet3d=>loc3_downsample.size():{}'.format(loc3_downsample.size()), color_idx=2)
        e3 = self.relu(self.encoder3(e2 + loc3_downsample))
        network_log('linknet3d=>e3.size():{}'.format(e3.size()), color_idx=2)

        '''
        loc4_downsample = self.downsample4(e3_fusion)
        network_log('linknet3d=>loc4_downsample.size():{}'.format(loc4_downsample.size()), color_idx=2)
        e4 = self.relu(self.encoder4(e3_fusion - loc4_downsample))
        network_log('[EB4]\nlinknet3d=>e4.size():{}'.format(e4.size()), color_idx=2)



        d4 = self.decoder4(e4)
        network_log('linknet3d=>d4.size():{}'.format(d4.size()), color_idx=1)

        d4_fusion = self.fusion(d4, e3)
        network_log('linknet3d=>d4_cat.size():{}'.format(d4_fusion.size()), color_idx=1)
        '''

        d3 = self.decoder3(e3)
        network_log('linknet3d=>d3.size():{}'.format(d3.size()), color_idx=1)

        d3_fusion = d3 + e2
        network_log('linknet3d=>d3_cat.size():{}'.format(d3_fusion.size()), color_idx=1)
        d2 = self.decoder2(d3_fusion)
        network_log('linknet3d=>d2.size():{}'.format(d2.size()), color_idx=1)

        d2_fusion = d2 + e1
        network_log('linknet3d=>d2_cat.size():{}'.format(d2_fusion.size()), color_idx=1)
        d1 = self.decoder1(d2_fusion)
        network_log('linknet3d=>d1.size():{}'.format(d1.size()), color_idx=1)


        f1 = self.finaldeconvbnrelu1(d1)
        network_log('linknet3d=>f1.size():{}'.format(f1.size()), color_idx=2)
        f2 = self.finalconvbnrelu2(f1)
        network_log('linknet3d=>f2.size():{}'.format(f2.size()), color_idx=2)
        f3 = self.finalconv3(f2)
        network_log('linknet3d=>f3.size():{}'.format(f3.size()), color_idx=2)


        #mb0 = F.max_pool3d(input1_downsample, kernel_size=input1_downsample.size()[2:])
        mb1 = F.max_pool3d(loc1_downsample, kernel_size=loc1_downsample.size()[2:])
        mb2 = F.max_pool3d(loc2_downsample, kernel_size=loc2_downsample.size()[2:])
        #mb3 = F.max_pool3d(loc3_downsample, kernel_size=loc3_downsample.size()[2:])
        #mb4 = F.max_pool3d(loc4_downsample, kernel_size=loc4_downsample.size()[2:])
        #mb_fusion = torch.cat([mb1, mb2, mb3, mb4], dim=1)
        mb_fusion = torch.cat([mb1, mb2], dim=1)
        #mb_fusion = torch.cat([mb1, mb2, mb3], dim=1)
        
        mb_fusion_dropout = self.dropout(mb_fusion)
        mb_fusion_flatten = mb_fusion_dropout.view(-1, mb_fusion_dropout.size()[1])
        mb_final = self.linear(mb_fusion_flatten)
        #print( mb1.size(), mb2.size(), mb3.size(), mb_fusion_flatten.size(), mb_final.size())
        '''
        network_log('linknet3d=>input1_downsample.size():{}'.format(input1_downsample.size()), color_idx=1)
        network_log('linknet3d=>e1_fusion_downsample.size():{}'.format(e1_fusion_downsample.size()), color_idx=1)
        network_log('linknet3d=>e2_fusion_downsample.size():{}'.format(e2_fusion_downsample.size()), color_idx=1)
        network_log('linknet3d=>e3_fusion_downsample.size():{}'.format(e3_fusion_downsample.size()), color_idx=1)
        # splitline
        network_log('linknet3d=>inputs2.size():{}'.format(input2.size()), color_idx=1)
        mb1 = F.max_pool3d(e1, kernel_size=e1.size()[2:])
        network_log('linknet3d=>mb1.size():{}'.format(mb1.size()), color_idx=1)
        mb1_flatten = mb1.view(-1, mb1.size()[1])
        mb2 = F.max_pool3d(e2, kernel_size=e2.size()[2:])
        network_log('linknet3d=>mb2.size():{}'.format(mb2.size()), color_idx=1)
        mb2_flatten = mb2.view(-1, mb2.size()[1])
        mb3 = F.max_pool3d(e3, kernel_size=e3.size()[2:])
        network_log('linknet3d=>mb3.size():{}'.format(mb3.size()), color_idx=1)
        mb3_flatten = mb3.view(-1, mb3.size()[1])
        # mb4 = F.max_pool3d(e4, kernel_size=e4.size()[2:])
        # mb4_flatten = mb4.view(-1, mb4.size()[1])
        '''
        return f3, mb_final