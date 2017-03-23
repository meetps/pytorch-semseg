import torch
import numpy as np
import torch.nn as nn
import scipy.misc as m 
import torch.nn.functional as F

from torch.autograd import Variable

from ptsemseg.models.segnet import segnet 
from ptsemseg.models.fcn import fcn32s
from ptsemseg.models.unet import unet

'''
Global Parameters
'''
img_rows    = 224
img_cols    = 224
batch_size  = 1
n_epoch     = 10000
n_classes   = 12
l_rate      = 0.0001

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
data_path = '/home/gpu_users/meetshah/camvid'


def getRandomIdx(n_samples, batch_size):
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    return idx[0:batch_size]


def train():

    model = unet(n_classes=n_classes, is_batchnorm=True, in_channels=3, is_deconv=True)
    if torch.cuda.is_available():
        model.cuda(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    x_train = np.load(data_path + 'x_train.npy')
    y_train = np.argmax(np.load(data_path + 'y_train.npy'), axis=3)

    for epoch in range(n_epoch):
        idx = getRandomIdx(x_train.shape[0], batch_size)
        images = x_train[idx]
        labels = y_train[idx]

        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.resize_(batch_size, img_cols*img_cols, n_classes)
        labels = labels.permute(0, 2, 3, 1)
        labels = labels.resize_(batch_size, img_cols*img_cols, n_classes)

        loss = F.cross_entropy(outputs, labels, class_weighting)

        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epoch, loss.data[0]))

        if (epoch+1) % 20 == 0:
            l_rate /= 3
            optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    torch.save(model, "unet_camvid.pkl")

if __name__ == '__main__':
    train()