import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models.segnet import segnet 
from ptsemseg.models.fcn import fcn32s
from ptsemseg.models.unet import unet
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader


'''
Global Parameters
'''
img_rows      = 224
img_cols      = 224
batch_size    = 1
n_epoch       = 10000
n_classes     = 21
l_rate        = 0.001
feature_scale = 1

data_path = '/home/gpu_users/meetshah/segdata/pascal/VOCdevkit/VOC2012'


def train(model):

    if model == 'unet':
        model = unet(feature_scale=feature_scale,
                     n_classes=n_classes,
                     is_batchnorm=True,
                     in_channels=3,
                     is_deconv=True)

    if model == 'segnet':
        model = segnet(n_classes=n_classes,
                       in_channels=3,
                       is_unpooling=True)

    camVid = pascalVOCLoader(data_path, is_transform=True, img_size=img_rows)
    trainloader = data.DataLoader(camVid, batch_size=batch_size, num_workers=4)

    if torch.cuda.is_available():
        model.cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)

    for epoch in range(n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)

            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.resize(img_cols*img_rows, n_classes)
            labels = labels.resize(img_cols*img_rows)

            loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epoch, loss.data[0]))

        # if (epoch+1) % 20 == 0:
            # l_rate /= 3
            # optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    torch.save(model, "unet_voc_" + str(feature_scale) + ".pkl")

if __name__ == '__main__':
    model = sys.argv[1]
    train(model)