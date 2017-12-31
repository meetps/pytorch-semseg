import os
import sys
import torch
import visdom
import argparse
import skvideo.io
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

def image_blend(img1, img2, alpha):
    img_new = np.zeros(img1.size, dtype=np.float32)
    img_new = img1 * alpha + img2 * (1 - alpha)
    return img_new

def test_frame(args, frame, model, loader):

    img = frame
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (512, 1024))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    if torch.cuda.is_available():
        images = Variable(img.cuda(0), volatile=True)
    else:
        images = Variable(img, volatile=True)

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    return decoded * 255.0

def test(args):

    vid_seq = '02'
    path = '../leftImg8bit/demoVideo/stuttgart_{}/'.format(vid_seq)
    images_seq = [path + fn for fn in os.listdir(path)]
    images_seq.sort()

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    # Setup Model
    model = get_model(args.model_path[:args.model_path.find('_')], n_classes)   
    state = convert_state_dict(torch.load(args.model_path)['model_state'])      
    model.load_state_dict(state)                                                
    model.eval()                                                        

    if torch.cuda.is_available():
        model.cuda(0)

    ims = [misc.imresize(misc.imread(f), (512,1024)) for f in images_seq]


    alpha = 0.3
    writer = skvideo.io.FFmpegWriter("outputvideo{}_{}.mp4".format(vid_seq, alpha))
    for i, frame in enumerate(tqdm(ims)):
        overlay = test_frame(args, frame, model, loader)
        tf = np.array(image_blend(frame, overlay, alpha), dtype=np.uint8)
        writer.writeFrame(tf)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)

