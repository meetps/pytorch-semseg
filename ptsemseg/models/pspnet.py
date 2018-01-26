import torch
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
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData']

        def _get_layer_params(layer, ltype):

            if ltype == 'BNData':
                n_channels = layer.blobs[0].shape.dim[1]
                mean = np.array([w for w in layer.blobs[0].data]).reshape(n_channels)
                var = np.array([w for w in layer.blobs[1].data]).reshape(n_channels)
                scale_factor = np.array([w for w in layer.blobs[2].data]).reshape(n_channels)
                mean, var = mean / scale_factor[0], var / scale_factor[0]
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

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())
            
            np.testing.assert_array_equal(weights.shape, w_shape)
            print("CONV: Original {} and trans weights {}".format(w_shape,
                                                                  weights.shape))
            module.weight.data = torch.from_numpy(weights)

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                np.testing.assert_array_equal(bias.shape, b_shape)
                print("CONV: Original {} and trans bias {}".format(b_shape,
                                                                   bias.shape))
                module.bias.data = torch.from_numpy(bias)

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]
            
            _transfer_conv(conv_layer_name, conv_module)
            
            mean, var, _ = layer_params[conv_layer_name+'/bn']
            print("BN: Original {} and trans weights {}".format(bn_module.running_mean.size(),
                                                                mean.shape))
            bn_module.running_mean = torch.from_numpy(mean)
            bn_module.running_var = torch.from_numpy(var)
            
        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]

            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {prefix + '_1_1x1_reduce': bottleneck.cbr1.cbr_unit,
                                      prefix + '_1_3x3': bottleneck.cbr2.cbr_unit,
                                      prefix + '_1_1x1_proj': bottleneck.cb4.cb_unit,
                                      prefix + '_1_1x1_increase': bottleneck.cb3.cb_unit,}

            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)

            for layer_idx in range(2, n_layers+1):
                residual_layer = block_module.layers[layer_idx-1]
                residual_conv_bn_dic = {'_'.join(map(str, [prefix, layer_idx, '1x1_reduce'])): residual_layer.cbr1.cbr_unit,
                                        '_'.join(map(str, [prefix, layer_idx, '3x3'])):  residual_layer.cbr2.cbr_unit,
                                        '_'.join(map(str, [prefix, layer_idx, '1x1_increase'])): residual_layer.cb3.cb_unit,} 
                
                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)


        convbn_layer_mapping = {'conv1_1_3x3_s2': self.convbnrelu1_1.cbr_unit,
                                'conv1_2_3x3': self.convbnrelu1_2.cbr_unit,
                                'conv1_3_3x3': self.convbnrelu1_3.cbr_unit,
                                'conv5_3_pool6_conv': self.pyramid_pooling.paths[0].cbr_unit, 
                                'conv5_3_pool3_conv': self.pyramid_pooling.paths[1].cbr_unit,
                                'conv5_3_pool2_conv': self.pyramid_pooling.paths[2].cbr_unit,
                                'conv5_3_pool1_conv': self.pyramid_pooling.paths[3].cbr_unit,
                                'conv5_4': self.cbr_final.cbr_unit,}

        residual_layers = {'conv2': [self.res_block2, self.block_config[0]],
                           'conv3': [self.res_block3, self.block_config[1]],
                           'conv4': [self.res_block4, self.block_config[2]],
                           'conv5': [self.res_block5, self.block_config[3]],}

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        _transfer_conv('conv6', self.classification)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

if __name__ == '__main__':
    from torch.autograd import Variable
    psp = pspnet(n_classes=19)
    psp.load_pretrained_model(model_path='/home/meetshah1995/models/pspnet101_cityscapes.caffemodel')
    psp.float()
    inp = Variable(torch.ones([1, 3, 713, 713]))
    out = psp(inp)
    print("Output Shape {} \t Input Shape {}".format(out.size(), inp.size()))