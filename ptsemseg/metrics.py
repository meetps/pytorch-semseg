# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.patch_dice_list = []
        self.macroblock_labels = []
        self.macroblock_predicts = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds, macroblock_labels, macroblock_predicts):
        self.macroblock_labels.extend(macroblock_labels)
        self.macroblock_predicts.extend(macroblock_predicts)
        for data_idx in range(label_preds.shape[0]):
            label_pred, label_true = label_preds[data_idx], label_trues[data_idx]
            dice = np.sum(label_pred[label_true == 1]) * 2.0 / (np.sum(label_pred) + np.sum(label_true))
            self.patch_dice_list.append(dice)
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc    : \t": acc,
                "Mean    Acc    : \t": acc_cls,
                "FreqW   Acc    : \t": fwavacc,
                "Mean IoU       : \t": mean_iu,
                "Patch DICE LIST: \t": self.patch_dice_list,
                "Patch DICE AVER: \t": sum(self.patch_dice_list)/len(self.patch_dice_list),
                "MacroBlock_lbls: \t": self.macroblock_labels,
                "MacroBlock_prds: \t": self.macroblock_predicts
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.patch_dice_list = []
        self.macroblock_labels = []
        self.macroblock_predicts = []



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

