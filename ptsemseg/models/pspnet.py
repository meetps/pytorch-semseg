import numpy as np
import torch.nn as nn

from ptsemseg import caffe_pb2
from ptsemseg.models.utils import *


class pspnet(nn.Module):

    def __init__(self, n_classes=21, block_config=[3, 4, 23, 3]):
        super(pspnet, self).__init__()
        
        self.block_config = block_config
        self.n_classes = n_classes

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=64,
                                                 padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)
        
        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.classification = nn.Conv2d(512, n_classes, 1, 1, 0)

    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_3(self.convbnrelu1_2(self.convbnrelu1_1(x)))
        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)
        # H/4, W/4 -> H/8, W/8
        x = self.res_block5(self.res_block4(self.res_block3(self.res_block2(x))))
        x = self.pyramid_pooling(x)
        x = F.dropout2d(self.cbr_final(x), p=0.1, inplace=True)
        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')
        return x

    def load_pretrained_model(self, model_path):
        """
        Done: Load weights from caffemodel w/o caffe dependency
        TODO: Plug them in corresponding modules
        """

        # Only care about layer_types that have trainable parameters
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData'] 

        def _get_layer_params(layer, ltype):

            if ltype == 'BNData':
                n_channels = layer.blobs[0].shape.dim[1]
                mean = np.array([w for w in layer.blobs[0].data]).reshape(n_channels)
                var = np.array([w for w in layer.blobs[1].data]).reshape(n_channels)
                scale_factor = np.array([w for w in layer.blobs[2].data]).reshape(n_channels)
                mean, var = mean / scale_factor, var / scale_factor
                return [mean, var, scale_factor]

            elif ltype in ['ConvolutionData', 'HoleConvolutionData']:
                is_bias = layer.convolution_param.bias_term
                shape = [int(d) for d in layer.blobs[0].shape.dim]
                weights = np.array([w for w in layer.blobs[0].data]).reshape(shape)
                bias = []
                if is_bias:
                    bias = np.array([w for w in layer.blobs[1].data]).reshape(shape[0])
                return [weights, bias]
            
            elif ltype == 'InnerProduct':
                raise Exception("Fully connected layers {}, not supported".format(ltype))

            else:
                raise Exception("Unkown layer type {}".format(ltype))


        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                print("Processing layer {}".format(lname))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        #TODO: Plug weights from dictionary into right places

if __name__ == '__main__':
    psp = pspnet()
    psp.load_pretrained_model(model_path='/home/meetshah1995/models/pspnet101_cityscapes.caffemodel')
    import pdb;pdb.set_trace()