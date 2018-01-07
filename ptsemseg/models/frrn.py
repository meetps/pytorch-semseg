import torch.nn as nn
import torch.nn.functional as F
import functools

from ptsemseg.models.utils import *
from ptsemseg.loss import bootstrapped_cross_entropy2d

frrn_specs_dic = {
    'A': 
    {
        'encoder': [[3, 96, 2],
                    [4, 192, 4],
                    [2, 384,  8],
                    [2, 384,  16]],
        
        'decoder':  [[2, 192, 8],
                     [2, 192, 4],
                     [2, 96,  2]],
    },

    'B': 
    {
        'encoder': [[3, 96, 2],
                    [4, 192, 4],
                    [2, 384, 8],
                    [2, 384, 16],
                    [2, 384, 32]],
        
        'decoder':  [[2, 192, 16],
                     [2, 192, 8],
                     [2, 192, 4],
                     [2, 96,  2]],
    },}

class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323

    References: 
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=21, model_type=None):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.K = 64 * 512
        self.loss = functools.partial(bootstrapped_cross_entropy2d, K=self.K)

        self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48, kernel_size=3, strides=1))
            self.down_residual_units.append(RU(channels=48, kernel_size=3, strides=1))

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(48, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']
        
        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:    
            for block in range(n_blocks):
                key = '_'.join(map(str,['encoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str,['decoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        self.merge_conv = nn.Conv2d(prev_channels+32, 48,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        self.classif_conv = nn.Conv2d(48, self.n_classes,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)


    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv(x)

        prev_channels = 48
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str,['encoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s*2 for _s in y.size()[-2:]]) 
            y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear')
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str,['decoding_frru', n_blocks, channels, scale, block]))
                #print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                #print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([F.upsample(y, scale_factor=2, mode='bilinear' ), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)
        
        return x

