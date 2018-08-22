import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import *


class refinenet(nn.Module):
    """
    RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
    URL: https://arxiv.org/abs/1611.06612

    References: 
    1) Original Author's MATLAB code: https://github.com/guosheng/refinenet 
    2) TF implementation by @eragonruan: https://github.com/eragonruan/refinenet-image-segmentation
    """

    def __init__(self, n_classes=21):
        super(refinenet, self).__init__()
        self.n_classes = n_classes

        pass

    def forward(self, x):
        pass
