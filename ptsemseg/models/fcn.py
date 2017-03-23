import torch.nn as nn


class fcn32s(nn.Module):

    def __init__(self, n_classses=21, learned_billinear=True):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(

            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, n_classes, 1),
        )

        if self.learned_billinear:
            upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, stride=32, bias=False)
        else:
            upscore = nn.UpsamplingBilinear2d(scale_factor=32)
            upscore.scale_factor = None
        
        self.upscore = nn.Sequential(upscore)

    def forward(self, x):
        h = self.features(x)

        h = self.classifier(h)

        if self.use_deconv:
            h = self.upscore(h)
        else:
            from chainer.utils import conv
            in_h, in_w = h.size()[2:4]
            out_h = conv.get_deconv_outsize(in_h, k=64, s=32, p=0)
            out_w = conv.get_deconv_outsize(in_w, k=64, s=32, p=0)
            self.upscore[0].size = out_h, out_w
            h = self.upscore(h)
        h = h[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()
        return h