import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models.segnet import segnet 
from ptsemseg.models.fcn import fcn32s, fcn16s, fcn8s
from ptsemseg.models.unet import unet
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loss import cross_entropy2d

'''
Global Parameters
'''
img_rows      = 224
img_cols      = 224
batch_size    = 1
n_epoch       = 10000
n_classes     = 21
l_rate        = 1e-10
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

    pascal = pascalVOCLoader(data_path, is_transform=True, img_size=img_rows)
    trainloader = data.DataLoader(pascal, batch_size=batch_size, num_workers=4)

    if torch.cuda.is_available():
        model.cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate, momentum=0.99, weight_decay=5e-4)

    test_image, test_segmap = pascal[0]
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

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epoch, loss.data[0]))

        test_output = model(test_image)
        predicted = pascal.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        target = pascal.decode_segmap(test_segmap.numpy())

        vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))


    torch.save(model, "unet_voc_" + str(feature_scale) + ".pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))
    
    model = sys.argv[1]
    train(model)