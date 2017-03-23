import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, k_size, n_filters, padding):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, padding=padding),
                                 nn.BatchNorm2d(n_filters),)

        def forward(self, inputs):
            outputs = self.cb_unit(inputs)
            return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, k_size, n_filters, padding):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, padding=padding),
                                 nn.BatchNorm2d(n_filters),
                                 nn.ReLU(inplace=True),)

        def forward(self, inputs):
            outputs = self.cbr_unit(inputs)
            return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv2(self.conv1(inputs))
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, 2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2*[offset // 2, offset // 2 + 1]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class segnetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown, self).__init__()
        self.conv = conv2DBatchNormRelu(in_size, 3, out_size, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 1, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices


class segnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 1)
        self.conv = conv2DBatchNorm(in_size, 3, out_size, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs, indices = self.conv(outputs)
        return outputs, indices