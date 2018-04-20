import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True

cudnn.benchmark = True

def validate(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4)
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    model.cuda()

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = Variable(images.cuda(), volatile=True)
        #labels = Variable(labels.cuda(), volatile=True)

        if args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = Variable(torch.from_numpy( flipped_images ).float().cuda(), volatile=True)
            outputs_flipped = model( flipped_images )
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        #gt = labels.data.cpu().numpy()
        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print('Inference time (iter {0:5d}): {1:3.5f} fps'.format(i+1, pred.shape[0]/elapsed_time))
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--eval_flip', dest='eval_flip', action='store_true', 
                        help='Enable evaluation with flipped image | True by default')
    parser.add_argument('--no-eval_flip', dest='eval_flip', action='store_false', 
                        help='Disable evaluation with flipped image | True by default')
    parser.set_defaults(eval_flip=True)

    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')

    parser.add_argument('--measure_time', dest='measure_time', action='store_true', 
                        help='Enable evaluation with time (fps) measurement | True by default')
    parser.add_argument('--no-measure_time', dest='measure_time', action='store_false', 
                        help='Disable evaluation with time (fps) measurement | True by default')
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()
    validate(args)
