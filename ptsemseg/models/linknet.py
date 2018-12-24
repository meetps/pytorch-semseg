import torch.nn as nn

from ptsemseg.models.utils import conv2DBatchNormRelu, linknetUp, residualBlock


class linknet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(linknet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.layers = [2, 2, 2, 2]  # Currently hardcoded for ResNet-18

        filters = [64, 128, 256, 512]
        filters = [x / self.feature_scale for x in filters]

        self.inplanes = filters[0]

        # Encoder
        self.convbnrelu1 = conv2DBatchNormRelu(
            in_channels=3, k_size=7, n_filters=64, padding=3, stride=2, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = residualBlock
        self.encoder1 = self._make_layer(block, filters[0], self.layers[0])
        self.encoder2 = self._make_layer(block, filters[1], self.layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], self.layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        # Decoder
        self.decoder4 = linknetUp(filters[3], filters[2])
        self.decoder4 = linknetUp(filters[2], filters[1])
        self.decoder4 = linknetUp(filters[1], filters[0])
        self.decoder4 = linknetUp(filters[0], filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32 / feature_scale, 3, 2, 1),
            nn.BatchNorm2d(32 / feature_scale),
            nn.ReLU(inplace=True),
        )
        self.finalconvbnrelu2 = conv2DBatchNormRelu(
            in_channels=32 / feature_scale,
            k_size=3,
            n_filters=32 / feature_scale,
            padding=1,
            stride=1,
        )
        self.finalconv3 = nn.Conv2d(32 / feature_scale, n_classes, 2, 2, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
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
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)

        return f3
