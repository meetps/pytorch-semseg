DEBUG=False
def log(s):
    if DEBUG:
        print(s)
##################
def display(string):
    print(string)
    logger.info(string)
###################
import math
from glob import glob
import random
def init_data_split_miccai2008(root):
    # init_train_val_split
    #file_paths = glob(root + '*_lesion*.nhdr')  # preprocessing_incompleted
    file_paths = glob(root + '*_lesion*_preprocessed.npy')  # preprocessing_completed
    ratio = 0.3
    case_index = [path.split('/')[-1].split('_lesion')[0] for path in file_paths]
    random.shuffle(case_index)
    case_index_UNC = [temp for temp in case_index if 'UNC' in temp]
    case_index_CHB = [temp for temp in case_index if 'CHB' in temp]
    random.shuffle(case_index_CHB)
    random.shuffle(case_index_UNC)
    val_case_index_UNC = case_index_UNC[:int(ratio * (len(case_index_UNC)))]
    val_case_index_CHB = case_index_CHB[:int(ratio * (len(case_index_CHB)))]
    val_case_index = []
    val_case_index.extend(val_case_index_CHB)
    val_case_index.extend(val_case_index_UNC)
    return {'file_paths': file_paths, 'val_case_index': val_case_index, 'case_index':case_index}
def init_data_split_sasha(root):
    file_paths = glob(root + '*GT_preprocessed.npy')
    ratio = 0.15
    case_index_withTP = [path.split('/')[-1].split('-GT')[0] for path in file_paths]
    case_index_noTP = list(set([path.split('/')[-1].split('_')[0] for path in file_paths]))
    random.shuffle(case_index_noTP)
    val_case_index_noTP = case_index_noTP[:int(ratio * (len(case_index_noTP)))]
    val_case_index_withTP = []
    for case_index in case_index_withTP:
        for val_case_index in val_case_index_noTP:
            if val_case_index in case_index:
                val_case_index_withTP.append(case_index)
                break
    val_case_index_withTP.sort()
    case_index_withTP.sort()
    return {'file_paths': file_paths, 'val_case_index': val_case_index_withTP, 'case_index':case_index_withTP}
###################
def prep_class_val_weights(ratio):
    weight_foreback = torch.ones(2)
    weight_foreback[0] = 1 / (1 - ratio)
    weight_foreback[1] = 1 / ratio
    weight_foreback = weight_foreback.cuda()
    display("CE's Weight:{}".format(weight_foreback))
    return weight_foreback
###################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
##################
import os
import subprocess
import sys
import yaml
import time
import shutil
import torch
import visdom
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations3d
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def train(cfg, writer, logger):
    # Setup dataset split before setting up the seed for random
    if cfg['data']['dataset'] == 'miccai2008':
        split_info = init_data_split_miccai2008(cfg['data']['path'])  # miccai2008 dataset
    elif cfg['data']['dataset'] == 'sasha':
        split_info = init_data_split_sasha(cfg['data']['path'])  # miccai2008 dataset

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Cross Entropy Weight
    weight = prep_class_val_weights(cfg['training']['cross_entropy_ratio'])

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    print(('augmentations_cfg:', augmentations))
    data_aug = get_composed_augmentations3d(augmentations)


    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug,
        split_info = split_info, patch_size = cfg['training']['patch_size'],
        mods = cfg['data']['mods'],
        macroblock_num_along_one_dim=cfg['data']['macroblock_num_along_one_dim'])

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        split_info = split_info, patch_size = cfg['training']['patch_size'],
        mods = cfg['data']['mods'],
        macroblock_num_along_one_dim=cfg['data']['macroblock_num_along_one_dim'])

    n_classes = t_loader.n_classes
    n_macroblocks = t_loader.n_macroblocks
    print('NumOfMacroBlocks:{}'.format(n_macroblocks))
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=False)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes, n_macroblocks).to(device)
    model.apply(weights_init)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))
    softmax_function = nn.Softmax(dim=1)
    criterion_mb = nn.CrossEntropyLoss()

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_total_meter = averageMeter()
    val_loss_seg_meter = averageMeter()
    val_loss_loc_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i_train_iter = start_iter

    display('Training from {}th iteration\n'.format(i_train_iter))
    while i_train_iter < cfg['training']['train_iters']:
        i_batch_idx = 0
        train_iter_start_time = time.time()
        for (images, labels, case_index_list, macroblock_labels) in trainloader:
            start_ts_network = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            macroblock_labels = macroblock_labels.to(device)

            optimizer.zero_grad()
            (outputs_FM, outputs_mb) = model(images)

            #print('Unique on labels:{}'.format(np.unique(labels.data.cpu().numpy())))    #[0, 1]
            #print('Unique on outputs:{}'.format(np.unique(outputs_FM.data.cpu().numpy())))  #[-1.15, +0.39]
            log('TrainIter=> images.size():{} labels.size():{} | outputs.size():{}'.format(images.size(), labels.size(), outputs_FM.size()))
            loss_seg = cfg['training']['loss_balance_ratio'] * loss_fn(input=outputs_FM, target=labels, weight=weight, size_average=cfg['training']['loss']['size_average']) #Input:FM, Softmax is built with crossentropy loss fucntion
            loss_loc = criterion_mb(outputs_mb, macroblock_labels)
            loss_total = loss_seg + loss_loc

            loss_total.backward()
            optimizer.step()
            
            time_meter.update(time.time() - start_ts_network)

            print_per_batch_check = True if cfg['training']['print_interval_per_batch'] else i_batch_idx+1 == len(trainloader)
            if (i_train_iter + 1) % cfg['training']['print_interval'] == 0 and print_per_batch_check:
                fmt_str = "Iter [{:d}/{:d}::{:d}/{:d}]  [Loss: {:.4f} | Loss_seg: {:.4f} + Loss_loc: {:.4f}]  NetworkTime/Image: {:.4f}"
                print_str = fmt_str.format(i_train_iter + 1,
                                           cfg['training']['train_iters'],
                                           i_batch_idx+1, len(trainloader),
                                           loss_total.item(),loss_seg.item(),loss_loc.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                display(print_str)
                writer.add_scalar('loss/train_loss_total', loss_total.item(), i_train_iter+1)
                writer.add_scalar('loss/train_loss_seg', loss_seg.item(), i_train_iter + 1)
                writer.add_scalar('loss/train_loss_loc', loss_loc.item(), i_train_iter + 1)
                time_meter.reset()
            i_batch_idx += 1
        entire_time_all_cases = time.time()-train_iter_start_time
        display('EntireTime for {}th training iteration: {:.4f}   EntireTime/Image: {:.4f}'.format(i_train_iter+1,
                                                                                                 entire_time_all_cases,
                                                                                                 entire_time_all_cases/(len(trainloader)*cfg['training']['batch_size'])))

        validation_check = (i_train_iter + 1) % cfg['training']['val_interval'] == 0 or \
                           (i_train_iter + 1) == cfg['training']['train_iters']
        if not validation_check:
            print('')
        else:
            model.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val, case_index_list_val, macroblock_labels_val) in enumerate(valloader):
                    images_val = images_val.to(device)
                    labels_val = labels_val.to(device)
                    macroblock_labels_val = macroblock_labels_val.to(device)

                    (outputs_FM_val, outputs_mb_val) = model(images_val)
                    log('ValIter=> images_val.size():{} labels_val.size():{} | outputs.size():{}'.format(images_val.size(),
                                                                                                         labels_val.size(),
                                                                                                         outputs_FM_val.size()))#Input:FM, Softmax is built with crossentropy loss fucntion

                    val_loss_seg = cfg['training']['loss_balance_ratio'] * loss_fn(input=outputs_FM_val, target=labels_val, weight=weight, size_average=cfg['training']['loss']['size_average'])
                    val_loss_loc = criterion_mb(outputs_mb_val, macroblock_labels_val)
                    val_loss_total = val_loss_seg + val_loss_loc

                    outputs_CLASS_val = outputs_FM_val.data.max(1)[1]
                    outputs_mbCLASS_val = outputs_mb_val.data.max(1)[1]
                    outputs_PROB_val = softmax_function(outputs_FM_val.data)
                    outputs_lesionPROB_val = outputs_PROB_val[:, 1, :, :, :]

                    running_metrics_val.update(labels_val.data.cpu().numpy(), outputs_CLASS_val.cpu().numpy(),
                                               macroblock_labels_val.cpu().numpy(), outputs_mbCLASS_val.cpu().numpy())
                    val_loss_total_meter.update(val_loss_total.item())
                    val_loss_seg_meter.update(val_loss_seg.item())
                    val_loss_loc_meter.update(val_loss_loc.item())


                    '''
                        This FOR-LOOP is used to visualize validation data via tensorboard
                        It would take 3s roughly.
                    '''
                    for batch_identifier_index, case_index in enumerate(case_index_list_val):
                        tensor_grid = []
                        image_val = images_val[batch_identifier_index, :, :, :, :].float()               #torch.Size([3, 160, 160, 160])
                        label_val = labels_val[batch_identifier_index, :, :, :].float()                  #torch.Size([160, 160, 160])
                        output_lesionFM_val = outputs_FM_val[batch_identifier_index, 1, :, :, :].float()#torch.Size([160, 160, 160])
                        output_nonlesFM_val = outputs_FM_val[batch_identifier_index, 0, :, :, :].float()#torch.Size([160, 160, 160])
                        output_CLASS_val = outputs_CLASS_val[batch_identifier_index, :, :, :].float()   #torch.Size([160, 160, 160])
                        output_lesionPROB_val = outputs_lesionPROB_val[batch_identifier_index, :, :, :].float()    #torch.Size([160, 160, 160])
                        for z_index in range(images_val.size()[-1]):
                            label_slice = label_val[:, :, z_index]
                            output_CLASS_slice = output_CLASS_val[:, :, z_index]
                            if label_slice.sum() == 0 and output_CLASS_slice.sum() == 0:
                                continue

                            image_slice = image_val[:, :, :, z_index]
                            output_nonlesFM_slice = output_nonlesFM_val[:, :, z_index]
                            output_lesionFM_slice = output_lesionFM_val[:, :, z_index]
                            output_lesionPROB_slice = output_lesionPROB_val[:, :, z_index]

                            label_slice = F.pad(label_slice.unsqueeze_(0),(0,0,0,0,1,1))
                            output_CLASS_slice = F.pad(output_CLASS_slice.unsqueeze_(0),(0,0,0,0,2,0))
                            output_nonlesFM_slice = output_nonlesFM_slice.unsqueeze_(0).repeat(3, 1, 1)
                            output_lesionFM_slice = output_lesionFM_slice.unsqueeze_(0).repeat(3, 1, 1)
                            output_lesionPROB_slice = output_lesionPROB_slice.unsqueeze_(0).repeat(3, 1, 1)

                            slice_list = [image_slice, output_nonlesFM_slice, output_lesionFM_slice, output_lesionPROB_slice, output_CLASS_slice, label_slice]
                            #slice_list = [image_slice, output_lesionFM_slice, output_lesionPROB_slice, output_CLASS_slice, label_slice]
                            slice_grid = make_grid(slice_list, padding=20)
                            tensor_grid.append(slice_grid)
                        if len(tensor_grid) == 0:
                            continue
                        tensorboard_image_tensor = make_grid(tensor_grid, nrow=int(math.sqrt(len(tensor_grid)/6))+1, padding=0).permute(1, 2, 0).cpu().numpy()
                        writer.add_image(case_index, tensorboard_image_tensor, i_train_iter+1)
            writer.add_scalar('loss/val_loss_total', val_loss_total_meter.avg, i_train_iter+1)
            writer.add_scalar('loss/val_loss_seg', val_loss_seg_meter.avg, i_train_iter + 1)
            writer.add_scalar('loss/val_loss_loc', val_loss_loc_meter.avg, i_train_iter + 1)
            logger.info("Iter %d Loss_total: %.4f" % (i_train_iter + 1, val_loss_total_meter.avg))
            logger.info("Iter %d Loss_seg: %.4f" % (i_train_iter + 1, val_loss_seg_meter.avg))
            logger.info("Iter %d Loss_loc: %.4f" % (i_train_iter + 1, val_loss_loc_meter.avg))

            '''
                This CODE-BLOCK is used to calculate and update the evaluation matrcs 
            '''
            score, class_iou = running_metrics_val.get_scores()
            print('\x1b[1;32;44mValidationDataLoaded-EXPINDEX={}'.format(run_id))
            for k, v in score.items():
                print(k, v)
                logger.info('{}: {}'.format(k, v))
                if isinstance(v, list): continue
                writer.add_scalar('val_metrics/{}'.format(k), v, i_train_iter+1)

            for k, v in class_iou.items():
                print('IOU:cls_{}:{}'.format(k, v))
                logger.info('{}: {}'.format(k, v))
                writer.add_scalar('val_metrics/cls_{}'.format(k), v, i_train_iter+1)
            print('\x1b[0m\n')
            val_loss_total_meter.reset()
            val_loss_seg_meter.reset()
            val_loss_loc_meter.reset()
            running_metrics_val.reset()

            '''
                This IF-CHECK is used to update the best model
            '''
            if score["Mean IoU       : \t"] >= best_iou:
            #if score["Patch DICE AVER: \t"] >= best_iou:
                #best_iou = score["Patch DICE AVER: \t"]
                best_iou = score["Mean IoU       : \t"]

                state = {
                    "epoch": i_train_iter + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(writer.file_writer.get_logdir(),
                                         "{}_{}_best_model.pkl".format(
                                             cfg['model']['arch'],
                                             cfg['data']['dataset']))
                torch.save(state, save_path)
        i_train_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        #default="configs/miccai2008-anatomicalstructure.yml",
        default="configs/sasha.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    # Display Tensorboard
    print('TensorBoard::RUNDIR: {}'.format(logdir))

    # Display Config
    print('\x1b[1;33;40mCONFIG:')
    for key, value_dict in cfg.items():
        print('{}:'.format(key))
        for k, v in value_dict.items():
            print('\t\t{}: {}'.format(k, v))
    print('\x1b[0m')

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')
    subprocess.Popen("kill $(lsof -t -c tensorboa -a -i:6006)", shell=True)
    subprocess.Popen(['tensorboard', '--logdir', '{}'.format(logdir)])

    train(cfg, writer, logger)
