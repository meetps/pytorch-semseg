DEBUG=False
def log(s):
    if DEBUG:
        print(s)
###################
colors = ['\x1b[0m', '\x1b[6;30;42m', '\x1b[2;30;41m']
###################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample_bilinear as upsample_bilinear3d
from torch.autograd import Variable


class conv3DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv3DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv3d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv3d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm3d(int(n_filters)),
                                          nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
class conv3DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv3DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv3d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv3d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm3d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs
class residualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock3D, self).__init__()

        self.convbnrelu1 = conv3DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv3DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class deconv3DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv3DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose3d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm3d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs
class linknetUp3D(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(linknetUp3D, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W

        self.convbnrelu1 = conv3DBatchNormRelu(in_channels, n_filters / 2, k_size=1, stride=1, padding=0)

        # B, C/2, H, W -> B, C/2, 2H, 2W
        self.deconvbnrelu2 = deconv3DBatchNormRelu(n_filters / 2, n_filters / 2, k_size=2, stride=2, padding=0)

        # B, C/2, 2H, 2W -> B, C, 2H, 2W
        self.convbnrelu3 = conv3DBatchNormRelu(n_filters / 2, n_filters, k_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x

class unetConv3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv3d, self).__init__()
        padding = 1 # 0 before
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, 3, 1, padding),
                nn.BatchNorm3d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, 3, 1, padding),
                nn.BatchNorm3d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, padding), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, 3, 1, padding), nn.ReLU()
            )

    def forward(self, inputs):
        log('======>{} Conv3d=>inputs.size():{} {}'.format(colors[1], inputs.size(), colors[0]))
        outputs = self.conv1(inputs)
        log('======>{} Conv3d=>[After self.conv1()] inputs.size():{} {}'.format(colors[1], outputs.size(), colors[0]))
        outputs = self.conv2(outputs)
        log('======>{} Conv3d=>[After self.conv2()] inputs.size():{} {}'.format(colors[1], outputs.size(), colors[0]))
        return outputs

class unetUp3d(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp3d, self).__init__()
        padding = 0 # 0 before
        self.is_deconv = is_deconv
        self.conv = unetConv3d(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=padding)
    def forward(self, inputs1, inputs2):
        log('======>{} Up3d=>is_deconv:{} {}'.format(colors[2], self.is_deconv, colors[0]))
        log('======>{} Up3d=>inputs1.size():{}, inputs2.size():{} {}'.format(colors[2], inputs1.size(), inputs2.size(), colors[0]))
        if self.is_deconv:
            outputs2 = self.up(inputs2)
        else:
            outputs2 = upsample_bilinear3d(inputs2, scale_factor=2)
        log('======>{} Up3d=>[After-self.up()] inputs2.size():{} {}'.format(colors[2], outputs2.size(), colors[0]))
        '''
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, offset//2]
        log('======>{} Up3d=>[padding] offset:{} padding:{} {}'.format(colors[2], offset, padding, colors[0]))
        outputs1 = F.pad(inputs1, padding)
        '''
        outputs1 = inputs1
        log('======>{} Up3d=>[After-padding] inputs1.size():{} {}'.format(colors[2], outputs1.size(), colors[0]))

        cat_var = torch.cat([outputs1, outputs2], 1)
        log('======>{} Up3d=>[After-cat] outputs.size():{} {}'.format(colors[2], cat_var.size(), colors[0]))
        outputs = self.conv(cat_var)
        log('======>{} Up3d=>[After-conv] outputs.size():{} {}'.format(colors[2], outputs.size(), colors[0]))
        return outputs

