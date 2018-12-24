import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from ptsemseg import caffe_pb2
from ptsemseg.models.utils import (
    get_interp_size,
    cascadeFeatureFusion,
    conv2DBatchNormRelu,
    residualBlockPSP,
    pyramidPooling,
)
from ptsemseg.loss.loss import multi_scale_cross_entropy2d

icnet_specs = {
    "cityscapes": {"n_classes": 19, "input_size": (1025, 2049), "block_config": [3, 4, 6, 3]}
}


class icnet(nn.Module):

    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545

    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow

    """

    def __init__(
        self,
        n_classes=19,
        block_config=[3, 4, 6, 3],
        input_size=(1025, 2049),
        version=None,
        is_batchnorm=True,
    ):

        super(icnet, self).__init__()

        bias = not is_batchnorm

        self.block_config = (
            icnet_specs[version]["block_config"] if version is not None else block_config
        )
        self.n_classes = icnet_specs[version]["n_classes"] if version is not None else n_classes
        self.input_size = icnet_specs[version]["input_size"] if version is not None else input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(
            self.block_config[0], 64, 32, 128, 1, 1, is_batchnorm=is_batchnorm
        )
        self.res_block3_conv = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="conv",
            is_batchnorm=is_batchnorm,
        )
        self.res_block3_identity = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="identity",
            is_batchnorm=is_batchnorm,
        )

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(
            self.block_config[2], 256, 128, 512, 1, 2, is_batchnorm=is_batchnorm
        )
        self.res_block5 = residualBlockPSP(
            self.block_config[3], 512, 256, 1024, 1, 4, is_batchnorm=is_batchnorm
        )

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(
            1024, [6, 3, 2, 1], model_name="icnet", fusion_mode="sum", is_batchnorm=is_batchnorm
        )

        # Final conv layer with kernel 1 in sub4 branch
        self.conv5_4_k1 = conv2DBatchNormRelu(
            in_channels=1024,
            k_size=1,
            n_filters=256,
            padding=0,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu2_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu3_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.classification = nn.Conv2d(128, self.n_classes, 1, 1, 0)

        # Cascade Feature Fusion Units
        self.cff_sub24 = cascadeFeatureFusion(
            self.n_classes, 256, 256, 128, is_batchnorm=is_batchnorm
        )
        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 128, 64, 128, is_batchnorm=is_batchnorm
        )

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        h, w = x.shape[2:]

        # H, W -> H/2, W/2
        x_sub2 = F.interpolate(
            x, size=get_interp_size(x, s_factor=2), mode="bilinear", align_corners=True
        )

        # H/2, W/2 -> H/4, W/4
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)

        # H/4, W/4 -> H/8, W/8
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)

        # H/8, W/8 -> H/16, W/16
        x_sub2 = self.res_block2(x_sub2)
        x_sub2 = self.res_block3_conv(x_sub2)
        # H/16, W/16 -> H/32, W/32
        x_sub4 = F.interpolate(
            x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)

        x_sub4 = self.res_block4(x_sub4)
        x_sub4 = self.res_block5(x_sub4)

        x_sub4 = self.pyramid_pooling(x_sub4)
        x_sub4 = self.conv5_4_k1(x_sub4)

        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.convbnrelu3_sub1(x_sub1)

        x_sub24, sub4_cls = self.cff_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.cff_sub12(x_sub24, x_sub1)

        x_sub12 = F.interpolate(
            x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)
        sub124_cls = self.classification(x_sub12)

        if self.training:
            return (sub124_cls, sub24_cls, sub4_cls)
        else:
            sub124_cls = F.interpolate(
                sub124_cls,
                size=get_interp_size(sub124_cls, z_factor=4),
                mode="bilinear",
                align_corners=True,
            )
            return sub124_cls

    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = [
            "BNData",
            "ConvolutionData",
            "HoleConvolutionData",
            "Convolution",
        ]  # Convolution type for conv3_sub1_proj

        def _get_layer_params(layer, ltype):

            if ltype == "BNData":
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ["ConvolutionData", "HoleConvolutionData", "Convolution"]:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]

            elif ltype == "InnerProduct":
                raise Exception("Fully connected layers {}, not supported".format(ltype))

            else:
                raise Exception("Unkown layer type {}".format(ltype))

        net = caffe_pb2.NetParameter()
        with open(model_path, "rb") as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            lbottom = l.bottom
            ltop = l.top
            if ltype in ltypes:
                print("Processing layer {} | {}, {}".format(lname, lbottom, ltop))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)
            # if len(l.blobs) > 0:
            #    print(lname, ltype, lbottom, ltop, len(l.blobs))

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        # _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())

            print(
                "CONV {}: Original {} and trans weights {}".format(
                    layer_name, w_shape, weights.shape
                )
            )

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print(
                    "CONV {}: Original {} and trans bias {}".format(layer_name, b_shape, bias.shape)
                )
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_bn(conv_layer_name, bn_module):
            mean, var, gamma, beta = layer_params[conv_layer_name + "/bn"]
            print(
                "BN {}: Original {} and trans weights {}".format(
                    conv_layer_name, bn_module.running_mean.size(), mean.shape
                )
            )
            bn_module.running_mean.copy_(torch.from_numpy(mean).view_as(bn_module.running_mean))
            bn_module.running_var.copy_(torch.from_numpy(var).view_as(bn_module.running_var))
            bn_module.weight.data.copy_(torch.from_numpy(gamma).view_as(bn_module.weight))
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            _transfer_conv(conv_layer_name, conv_module)

            if conv_layer_name + "/bn" in layer_params.keys():
                bn_module = mother_module[1]
                _transfer_bn(conv_layer_name, bn_module)

        def _transfer_residual(block_name, block):
            block_module, n_layers = block[0], block[1]
            prefix = block_name[:5]

            if ("bottleneck" in block_name) or ("identity" not in block_name):  # Conv block
                bottleneck = block_module.layers[0]
                bottleneck_conv_bn_dic = {
                    prefix + "_1_1x1_reduce": bottleneck.cbr1.cbr_unit,
                    prefix + "_1_3x3": bottleneck.cbr2.cbr_unit,
                    prefix + "_1_1x1_proj": bottleneck.cb4.cb_unit,
                    prefix + "_1_1x1_increase": bottleneck.cb3.cb_unit,
                }

                for k, v in bottleneck_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)

            if ("identity" in block_name) or ("bottleneck" not in block_name):  # Identity blocks
                base_idx = 2 if "identity" in block_name else 1

                for layer_idx in range(2, n_layers + 1):
                    residual_layer = block_module.layers[layer_idx - base_idx]
                    residual_conv_bn_dic = {
                        "_".join(
                            map(str, [prefix, layer_idx, "1x1_reduce"])
                        ): residual_layer.cbr1.cbr_unit,
                        "_".join(
                            map(str, [prefix, layer_idx, "3x3"])
                        ): residual_layer.cbr2.cbr_unit,
                        "_".join(
                            map(str, [prefix, layer_idx, "1x1_increase"])
                        ): residual_layer.cb3.cb_unit,
                    }

                    for k, v in residual_conv_bn_dic.items():
                        _transfer_conv_bn(k, v)

        convbn_layer_mapping = {
            "conv1_1_3x3_s2": self.convbnrelu1_1.cbr_unit,
            "conv1_2_3x3": self.convbnrelu1_2.cbr_unit,
            "conv1_3_3x3": self.convbnrelu1_3.cbr_unit,
            "conv1_sub1": self.convbnrelu1_sub1.cbr_unit,
            "conv2_sub1": self.convbnrelu2_sub1.cbr_unit,
            "conv3_sub1": self.convbnrelu3_sub1.cbr_unit,
            # 'conv5_3_pool6_conv': self.pyramid_pooling.paths[0].cbr_unit,
            # 'conv5_3_pool3_conv': self.pyramid_pooling.paths[1].cbr_unit,
            # 'conv5_3_pool2_conv': self.pyramid_pooling.paths[2].cbr_unit,
            # 'conv5_3_pool1_conv': self.pyramid_pooling.paths[3].cbr_unit,
            "conv5_4_k1": self.conv5_4_k1.cbr_unit,
            "conv_sub4": self.cff_sub24.low_dilated_conv_bn.cb_unit,
            "conv3_1_sub2_proj": self.cff_sub24.high_proj_conv_bn.cb_unit,
            "conv_sub2": self.cff_sub12.low_dilated_conv_bn.cb_unit,
            "conv3_sub1_proj": self.cff_sub12.high_proj_conv_bn.cb_unit,
        }

        residual_layers = {
            "conv2": [self.res_block2, self.block_config[0]],
            "conv3_bottleneck": [self.res_block3_conv, self.block_config[1]],
            "conv3_identity": [self.res_block3_identity, self.block_config[1]],
            "conv4": [self.res_block4, self.block_config[2]],
            "conv5": [self.res_block5, self.block_config[3]],
        }

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        _transfer_conv("conv6_cls", self.classification)
        _transfer_conv("conv6_sub4", self.cff_sub24.low_classifier_conv)
        _transfer_conv("conv6_sub2", self.cff_sub12.low_classifier_conv)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

    def tile_predict(self, imgs, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.

        Strides are adaptively computed from the imgs shape
        and input size

        :param imgs: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param n_classes: int with number of classes in seg output.
        """

        side_x, side_y = self.input_size
        n_classes = self.n_classes
        n_samples, c, h, w = imgs.shape
        # n = int(max(h,w) / float(side) + 1)
        n_x = int(h / float(side_x) + 1)
        n_y = int(w / float(side_y) + 1)
        stride_x = (h - side_x) / float(n_x)
        stride_y = (w - side_y) / float(n_y)

        x_ends = [[int(i * stride_x), int(i * stride_x) + side_x] for i in range(n_x + 1)]
        y_ends = [[int(i * stride_y), int(i * stride_y) + side_y] for i in range(n_y + 1)]

        pred = np.zeros([n_samples, n_classes, h, w])
        count = np.zeros([h, w])

        slice_count = 0
        for sx, ex in x_ends:
            for sy, ey in y_ends:
                slice_count += 1

                imgs_slice = imgs[:, :, sx:ex, sy:ey]
                if include_flip_mode:
                    imgs_slice_flip = torch.from_numpy(
                        np.copy(imgs_slice.cpu().numpy()[:, :, :, ::-1])
                    ).float()

                is_model_on_cuda = next(self.parameters()).is_cuda

                inp = Variable(imgs_slice, volatile=True)
                if include_flip_mode:
                    flp = Variable(imgs_slice_flip, volatile=True)

                if is_model_on_cuda:
                    inp = inp.cuda()
                    if include_flip_mode:
                        flp = flp.cuda()

                psub1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1

                pred[:, :, sx:ex, sy:ey] = psub
                count[sx:ex, sy:ey] += 1.0

        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


# For Testing Purposes only
if __name__ == "__main__":
    cd = 0
    import os
    import scipy.misc as m
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl

    ic = icnet(version="cityscapes", is_batchnorm=False)

    # Just need to do this one time
    caffemodel_dir_path = "PATH_TO_ICNET_DIR/evaluation/model"
    ic.load_pretrained_model(
        model_path=os.path.join(caffemodel_dir_path, "icnet_cityscapes_train_30k.caffemodel")
    )
    # ic.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path,
    #                           'icnet_cityscapes_train_30k_bnnomerge.caffemodel'))
    # ic.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path,
    #                           'icnet_cityscapes_trainval_90k.caffemodel'))
    # ic.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path,
    #                           'icnet_cityscapes_trainval_90k_bnnomerge.caffemodel'))

    # ic.load_state_dict(torch.load('ic.pth'))

    ic.float()
    ic.cuda(cd)
    ic.eval()

    dataset_root_dir = "PATH_TO_CITYSCAPES_DIR"
    dst = cl(root=dataset_root_dir)
    img = m.imread(
        os.path.join(
            dataset_root_dir,
            "leftImg8bit/demoVideo/stuttgart_00/stuttgart_00_000000_000010_leftImg8bit.png",
        )
    )
    m.imsave("test_input.png", img)
    orig_size = img.shape[:-1]
    img = m.imresize(img, ic.input_size)  # uint8 with RGB mode
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([123.68, 116.779, 103.939])[:, None, None]
    img = np.copy(img[::-1, :, :])
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    out = ic.tile_predict(img)
    pred = np.argmax(out, axis=1)[0]
    pred = pred.astype(np.float32)
    pred = m.imresize(pred, orig_size, "nearest", mode="F")  # float32 with F mode
    decoded = dst.decode_segmap(pred)
    m.imsave("test_output.png", decoded)
    # m.imsave('test_output.png', pred)

    checkpoints_dir_path = "checkpoints"
    if not os.path.exists(checkpoints_dir_path):
        os.mkdir(checkpoints_dir_path)
    ic = torch.nn.DataParallel(ic, device_ids=range(torch.cuda.device_count()))
    state = {"model_state": ic.state_dict()}
    torch.save(state, os.path.join(checkpoints_dir_path, "icnet_cityscapes_train_30k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnetBN_cityscapes_train_30k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnet_cityscapes_trainval_90k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnetBN_cityscapes_trainval_90k.pth"))
    print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))
