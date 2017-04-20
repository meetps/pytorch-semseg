import sys
import torch
import visdom
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models.segnet import segnet 
from ptsemseg.models.fcn import fcn32s, fcn16s, fcn8s
from ptsemseg.models.unet import unet
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loss import cross_entropy2d

'''
Global Parameters
'''
img_rows      = 360
img_cols      = 480
batch_size    = 8
n_epoch       = 10000
n_classes     = 12
l_rate        = 0.0001
feature_scale = 1
use_weighing  = False

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
data_path = '/home/gpu_users/meetshah/camvid'


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

    if model == 'fcn32':
        model = fcn32s(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    if model == 'fcn16':
        model = fcn16s(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    if model == 'fcn8':
        model = fcn8s(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    camVid = camvidLoader(data_path, is_transform=True)
    trainloader = data.DataLoader(camVid, batch_size=batch_size, num_workers=4)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)

    test_image, test_segmap = camVid[0]
    test_image = Variable(test_image.unsqueeze(0).cuda(0))
    vis = visdom.Visdom()

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

            if use_weighing:
                weights = torch.Tensor(class_weighting).cuda(0)
                loss = cross_entropy2d(outputs, labels, weight=weights)
            else:
                loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epoch, loss.data[0]))


        # if (epoch+1) % 10 == 0:
        test_output = model(test_image)
        predicted = camVid.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        target = camVid.decode_segmap(test_segmap.numpy())

        vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))


    torch.save(model, "unet_camvid_" + str(feature_scale) + ".pkl")

if __name__ == '__main__':
    model = sys.argv[1]
    train(model)