import os
import sys
import yaml
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *


def train(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    data_aug = Compose([RandomRotate(10), RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = get_data_path(cfg['data']['dataset'])

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=cfg['training']['batch_size'], num_workers=8, shuffle=True
    )

    valloader = data.DataLoader(v_loader, batch_size=cfg['training']['batch_size'], num_workers=8)

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup visdom for visualization
    if cfg['training']['visdom']:
        vis = visdom.Visdom()

        loss_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(
                xlabel="minibatches",
                ylabel="Loss",
                title="Training Loss",
                legend=["Loss"],
            ),
        )

    # Setup Model
    model = get_model(cfg['model']['arch'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Check if model has custom optimizer / loss
    if hasattr(model.module, "optimizer"):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg['training']['l_rate'], 
                                momentum=cfg['training']['momentum'],
                                weight_decay=cfg['training']['weight_decay']
        )

    if hasattr(model.module, "loss"):
        print("Using custom loss")
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iter = checkpoint["epoch"]
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(cfg['training']['resume']))

    best_iou = -100.0
    i = start_iter
    while i <= cfg['training']['train_iters']:
        for (images, labels) in trainloader:
            i += 1
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if cfg['training']['visdom']:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update="append",
                )

            if (i + 1) % cfg['training']['print_interval'] == 0:
                print(
                    "Iter [%d/%d] Loss: %.4f" % (i + 1, cfg['training']['train_iters'], loss.item())
                )

            if (i + 1) % cfg['training']['val_interval'] == 0:
                model.eval()
                for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                    images_val = images_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs = model(images_val)
                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()
                    running_metrics_val.update(gt, pred)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    torch.save(state, "{}_{}_best_model.pkl".format(cfg['model']['arch'],
                                                                    cfg['data']['dataset']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    train(cfg)
