import torch
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

N_CLASSES = 151

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(900, 2)

    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 900)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def decode_segmap(temp, plot=False):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, N_CLASSES):
        r[temp == l] = 10 * (l % 10)
        g[temp == l] = l
        b[temp == l] = 0

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = misc.imread(args.img_path)
    orig_size = img.shape[:-1]

    img = misc.imresize(img, (240, 240))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= np.array([104.00699, 116.66877, 122.67892])
    img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup model
    model = get_model({"arch":"fcn8s"}, N_CLASSES, version="mit_sceneparsing_benchmark")
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Setup classifier
    classifier = Classifier()
    classifier.eval()
    classifier.to(device)

    images = img.to(device)
    outputs = model(images)
    # outputs = F.avg_pool2d(outputs, 8) # Uncomment to see the real feature map being used.
    pred_raw = outputs.data.max(1)[1]
    pred = np.squeeze(pred_raw.cpu().numpy(), axis=0)

    turn_logit = classifier(pred_raw.type(torch.cuda.FloatTensor) / N_CLASSES)
    print(turn_logit.detach().cpu().numpy())

    decoded = decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    misc.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    infer(args)
