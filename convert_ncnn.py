"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import math
import numpy as np


class LayerParameter_ncnn(object):

    def __init__(self):
        self.type = ''
        self.param = []
        self.weights = []


def CopyTuple(param):
    if isinstance(param, tuple):
        return param
    elif isinstance(param, int):
        return param, param
    else:
        assert type(param)


def ty(ncnn_type):
    def f(_):
        layer = LayerParameter_ncnn()
        layer.type = ncnn_type
        return layer
    return f


def data(inputs):
    layer = LayerParameter_ncnn()
    layer.type = 'Input'

    input_shape = inputs.data.numpy().shape
    for dim in range(1, 4):
        if dim - 1 < len(input_shape):
            size = input_shape[dim]
        else:
            size = -233
        layer.param.append('%ld' % size)
    return layer


def Slice(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Slice'

    # """ ncnn only support slicing on channel dimension """
    # assert pytorch_layer.axis == 1

    layer.param = {}
    num_slice = len(pytorch_layer.slice_point) + 1
    slice_param = ('%d' % num_slice)
    prev_offset = 0
    for p in pytorch_layer.slice_point:
        offset = p
        slice_param += (',%d' % (offset - prev_offset))
        prev_offset = offset
    slice_param += (',%d' % -233)

    layer.param['-23300'] = slice_param
    layer.param['1'] = ('%d' % (pytorch_layer.axis - 1))

    return layer


def Split(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Split'

    return layer


def permute(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Permute'
    assert len(pytorch_layer.rev_dim_indices) == 4, len(pytorch_layer.rev_dim_indices)
    assert pytorch_layer.rev_dim_indices[0] == 0, pytorch_layer.rev_dim_indices[0]

    """ order_type details at src/layer/permute.cpp """
    h, w, c = pytorch_layer.rev_dim_indices[1], pytorch_layer.rev_dim_indices[2], pytorch_layer.rev_dim_indices[3]
    order_type = 0
    if c == 1 and h == 2 and w == 3:
        order_type = 0
    elif c == 1 and h == 3 and w == 2:
        order_type = 1
    elif c == 2 and h == 1 and w == 3:
        order_type = 2
    elif c == 2 and h == 3 and w == 1:
        order_type = 3
    elif c == 3 and h == 1 and w == 2:
        order_type = 4
    elif c == 3 and h == 2 and w == 1:
        order_type = 5

    layer.param.append('%d' % order_type)
    return layer


def flatten(pytorch_layer):
    """ Only support flatten view """
    total = 1
    for dim in pytorch_layer.old_size:
        total *= dim
    assert ((pytorch_layer.new_sizes[1] == total) or (pytorch_layer.new_sizes[1] == -1))

    layer = LayerParameter_ncnn()
    layer.type = "Flatten"
    return layer


def inner_product(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'InnerProduct'

    blobs_weight = pytorch_layer.state_dict()['weight'].cpu().numpy() 
    num_output = pytorch_layer.out_features
    layer.param.append('%d' % num_output)
    try:
        bias = pytorch_layer.state_dict()['bias'].cpu().numpy()
    except:
        bias = []
    if len(bias):
        layer.param.append('%d' % True)
        layer.param.append('%d' % blobs_weight.size)
        layer.weights.append(np.array([0.]))
        layer.weights.append(blobs_weight)
        layer.weights.append(bias)
    else:
        layer.param.append('%d' % False)
        layer.param.append('%d' % blobs_weight.size)
        layer.weights.append(np.array([0.]))
        layer.weights.append(blobs_weight)

    return layer


def concat(pytorch_layer):
    layer = LayerParameter_ncnn()
    axis = int(pytorch_layer.dim)
    layer.type = 'Concat'
    if (axis == 1):
        pass
    else:
        dim = axis - 1 if axis >= 1 else 0
        layer.param.append('%d' % dim)
    return layer


def spatial_convolution(pytorch_layer):
    layer = LayerParameter_ncnn()

    blobs_weight = pytorch_layer.state_dict()['weight'].cpu().numpy()
    assert len(blobs_weight.shape) == 4, blobs_weight.shape
    (nOutputPlane, nInputPlane, kH, kW) = blobs_weight.shape

    try:
        bias = pytorch_layer.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    padH = pytorch_layer.padding[0]
    padW = pytorch_layer.padding[1]
    dH = pytorch_layer.stride[0]
    dW = pytorch_layer.stride[1]
    dilation = pytorch_layer.dilation[0]
    groups = pytorch_layer.groups

    if False:#pytorch_layer.transposed:
        layer.type = 'Deconvolution'
        layer.param.append('%d' % nInputPlane)

        """ ncnn: Need to swap input dim and output dim """
        blobs_weight = np.swapaxes(blobs_weight, 0, 1)
    else:
        layer.type = 'Convolution'
        layer.param.append('%d' % nOutputPlane)

    assert kH == kW, [kH, kW]
    assert dH == dW, [dH, dW]
    assert padH == padW, [padH, padW]
    layer.param.append('%d' % kH)
    layer.param.append('%d' % dilation)
    layer.param.append('%d' % dH)
    layer.param.append('%d' % padH)

    if len(bias):
        layer.param.append('%d' % True)
        layer.param.append('%d' % blobs_weight.size)
        layer.weights.append(np.array([0.]))
        layer.weights.append(blobs_weight)
        layer.weights.append(bias)
    else:
        layer.param.append('%d' % False)
        layer.param.append('%d' % blobs_weight.size)
        layer.weights.append(np.array([0.]))
        layer.weights.append(blobs_weight)

    if groups != 1:
        layer.param.append('%d' % groups)
        layer.type += "DepthWise"

    return layer


def FillBilinear(ch, k):
    blob = np.zeros(shape=(ch, 1, k, k))

    """ Create bilinear weights in numpy array """
    bilinear_kernel = np.zeros([k, k], dtype=np.float32)
    scale_factor = (k + 1) // 2
    if k % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(k):
        for y in range(k):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)

    for i in range(ch):
        blob[i, 0, :, :] = bilinear_kernel
    return blob


def UpsampleBilinear(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Deconvolution'

    assert pytorch_layer.scale_factor[0] == pytorch_layer.scale_factor[1]
    factor = int(pytorch_layer.scale_factor[0])
    c = int(pytorch_layer.input_size[1])
    k = 2 * factor - factor % 2

    num_output = c
    kernel_size = k
    stride = factor
    pad = int(math.ceil((factor - 1) / 2.))
    dilation = 1
    # group = c
    # weight_filler = 'bilinear'
    bias_term = False

    layer.param.append('%d' % num_output)
    layer.param.append('%d' % kernel_size)
    layer.param.append('%d' % dilation)
    layer.param.append('%d' % stride)
    layer.param.append('%d' % pad)
    layer.param.append('%d' % bias_term)

    # learning_param = pb2.ParamSpec()
    # learning_param.lr_mult = 0
    # learning_param.decay_mult = 0
    # layer.param.extend([learning_param])

    """ init weight blob of filter kernel """
    blobs_weight = FillBilinear(c, k)
    layer.param.append('%d' % blobs_weight.size)
    layer.weights.append(np.array([0.]))
    layer.weights.append(blobs_weight)

    return layer


def CopyPoolingParameter(pytorch_layer, layer):

    padH, padW = CopyTuple(pytorch_layer.padding)
    kH, kW = CopyTuple(pytorch_layer.kernel_size)
    dH, dW = CopyTuple(pytorch_layer.stride)

    assert kH == kW, [kH, kW]
    assert dH == dW, [dH, dW]
    assert padH == padW, [padH, padW]
    layer.param.append('%d' % kH)
    layer.param.append('%d' % dH)

    # if pytorch_layer.ceil_mode is True:
    layer.param.append('%d' % padH)

    """ TODO: global_pooling? """
    layer.param.append('%d' % 0)


def MaxPooling(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Pooling'
    layer.param.append('%d' % 0)
    CopyPoolingParameter(pytorch_layer, layer)
    return layer


def AvgPooling(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Pooling'
    layer.param.append('%d' % 1)
    CopyPoolingParameter(pytorch_layer, layer)
    return layer


def dropout(pytorch_layer):
    layer = LayerParameter_ncnn()
    dropout_ratio = float(pytorch_layer.p)
    layer.type = 'Dropout'
    if abs(dropout_ratio - 0.5) < 1e-3:
        pass
    else:
        scale = 1.0 - dropout_ratio
        layer.param.append('%f' % scale)
    return layer


def elu(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'ELU'
    alpha = pytorch_layer.additional_args[0]
    layer.param.append('%f' % alpha)
    return layer


def ReLU(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'ReLU'
    layer.param.append('%f' % 0.0)
    return layer


def leaky_ReLU(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'ReLU'
    negative_slope = float(pytorch_layer.additional_args[0])
    layer.param.append('%f' % negative_slope)
    return layer


def PReLU(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'PReLU'

    blobs_weight = pytorch_layer.next_functions[1][0].variable.data.numpy()
    layer.param.append('%d' % blobs_weight.size)
    layer.weights.append(blobs_weight)
    return layer


def MulConst(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Power'
    layer.param.append('%f' % 1)
    layer.param.append('%f' % float(pytorch_layer.constant))
    layer.param.append('%f' % 0)
    return layer


def AddConst(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Power'
    layer.param.append('%f' % 1)
    layer.param.append('%f' % 1)
    """ Constant to add should be filled by hand, since not visible in autograd """
    layer.param.append('%f' % float('inf'))
    return layer


def softmax(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Softmax'
    """ TODO: axis """
    layer.param.append('%d' % 0)

    return layer


def Eltwise(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Eltwise'
    """ operation: 0=mul 1=add 2=max """
    layer.param.append('%d' % 1)
    """  TODO: coefficient  """
    return layer


def eltwise_max(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'Eltwise'
    """ operation: 0=mul 1=add 2=max """
    layer.param.append('%d' % 2)
    """  TODO: coefficient  """
    return layer


def negate(pytorch_layer):
    layer = LayerParameter_ncnn()
    layer.type = 'UnaryOp'
    """ Operation_NEG=1, more op details at src/layer/unaryop.h """
    layer.param.append('%d' % 1)
    return layer


def batchnorm(pytorch_layer):
    layer_bn = LayerParameter_ncnn()
    layer_bn.type = 'BatchNorm'

    layer_bn.param.append('%d' % pytorch_layer.running_mean.numpy().size)

    layer_bn.weights.append(np.ones(pytorch_layer.running_mean.numpy().shape))
    layer_bn.weights.append(pytorch_layer.running_mean.numpy())
    """ Add eps by hand for running_var in ncnn """
    running_var = pytorch_layer.running_var.numpy()
    running_var = running_var + pytorch_layer.eps
    layer_bn.weights.append(running_var)
    layer_bn.weights.append(np.zeros(pytorch_layer.running_mean.numpy().shape))

    layer_scale = LayerParameter_ncnn()
    layer_scale.type = 'Scale'

    blobs_weight = pytorch_layer.state_dict()['weight'].cpu().numpy() 
    try:
        bias = pytorch_layer.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    if len(bias):
        layer_scale.param.append('%d' % blobs_weight.size)
        layer_scale.param.append('%d' % True)
        layer_scale.weights.append(blobs_weight)
        layer_scale.weights.append(bias)
    else:
        layer_scale.param.append('%d' % blobs_weight.size)
        layer_scale.param.append('%d' % False)
        layer_scale.weights.append(blobs_weight)

    return [layer_bn, layer_scale]

def AdaptiveAvgPooling(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'AdaptiveAvgPooling'
    layer.param.append('%d' % chimodule.output_size[0])
    layer.param.append('%d' % chimodule.output_size[1])
#    CopyPoolingParameter(pytorch_layer, layer)
    return layer

def Embed(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'Embed'
    weight = chimodule.state_dict()['weight'].cpu().numpy()
    try:
        bias = chimodule.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    print(weight.shape[1])
    if len(bias):
#        print("***************************Embed_bias")
        layer.param.append('%d' % weight.shape[1])
        layer.param.append('%d' % weight.shape[0])
        layer.param.append('%d' % weight.size)
        layer.param.append('%d' % True)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)
        layer.weights.append(bias)
    else:
        layer.param.append('%d' % weight.shape[1])
        layer.param.append('%d' % weight.shape[0])
        layer.param.append('%d' % weight.size)
        layer.param.append('%d' % False)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)
    
    return layer

def Sigmoid(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'Sigmoid'	

    return layer

def LSTMCell(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'LSTMCell'
    weight_ih = chimodule.state_dict()['weight_ih'].cpu().numpy()
    weight_hh = chimodule.state_dict()['weight_hh'].cpu().numpy()

#    print(type(weight))
    try:
        bias_ih = chimodule.state_dict()['bias_ih'].cpu().numpy()
        bias_hh = chimodule.state_dict()['bias_hh'].cpu().numpy()
    except:
        bias_ih = []
        bias_hh = []
    if len(bias_ih) and len(bias_hh):
#        print("***************************lstm_bias")
        layer.param.append('%d' % chimodule.input_size)
        layer.param.append('%d' % chimodule.hidden_size)
        layer.param.append('%d' % True)
        layer.weights.append(weight_ih)
        layer.weights.append(weight_hh)
        layer.weights.append(bias_ih)
        layer.weights.append(bias_hh)
    else:
#        print("***************************lstm_nobias")
        layer.param.append('%d' % chimodule.input_size)
        layer.param.append('%d' % chimodule.hidden_size)
        layer.param.append('%d' % False)
        layer.weights.append(weight_ih)
        layer.weights.append(weight_hh)
    
    return layer

def look_for_convertlayer():
    return {
        'ReLU': ReLU,
        'Conv2d': spatial_convolution,
        'MaxPool2d': MaxPooling,
        'AdaptiveAvgPool2d': AdaptiveAvgPooling,
        'BatchNorm2d': batchnorm,
        'Linear': inner_product,
        'Embedding': Embed,
        'Softmax': softmax,
        'Dropout': dropout,
        'Sigmoid': Sigmoid,
        'LSTMCell': LSTMCell,
        'Add': Eltwise,
        'Split': Split,
    }

def link_ncnn(layer, name, bottom, top):
    if not layer:
        return
    layer_type = layer.type
    layer_param = layer.param
    if isinstance(layer_param, list):
        for ind, param in enumerate(layer_param):
            layer_param[ind] = str(ind) + '=' + param
    elif isinstance(layer_param, dict):
        param_dict = layer_param
        layer_param = []
        for key, param in param_dict.iteritems():
            layer_param.append(key + '=' + param)

    pp = []
    pp.append('%-16s' % layer_type)
    pp.append('%-16s %d %d' % (name, len(bottom), len(top)))
    for b in bottom:
        pp.append('%s' % b)
        if b not in blob_set:
            blob_set.add(b)
    for t in top:
        pp.append('%s' % t)
        if t not in blob_set:
            blob_set.add(t)
    layer_param = pp + layer_param

    ncnn_net.append(' '.join(layer_param))

    for w in layer.weights:
        ncnn_weights.append(w)

def convert_ncnn(inModule):
    """
    """
    from torchvision.models.resnet import Bottleneck
    global layer_type_count,top_name,bottom_name
    for chinam, chimodule in inModule.named_children():
        convert = look_for_convertlayer()
        if len(chimodule._modules) == 0:
            print(chinam)
#            print(chimodule._get_name(), 'func->', chimodule)
#            print('type', type(chimodule))
#            if 'Conv' in chimodule._get_name():
            layer_type_name = chimodule._get_name()
            if layer_type_name in layer_type_count:
                layer_type_count[layer_type_name] += 1
            else:
                layer_type_count[layer_type_name] = 1
            name = layer_type_name + '_' + str(layer_type_count[layer_type_name])
#            print(layer_type_name)
#            print(type(chimodule))
            layer = convert[layer_type_name](chimodule)
            if layer_type_name == 'BatchNorm2d':
                scale_name = name + '_' + 'scale'
                bottom_name = top_name
                top_name = [scale_name,]
                link_ncnn(layer[0], name, bottom_name, [name,])
                link_ncnn(layer[1], scale_name, [name,], top_name)
            else:
                bottom_name = top_name
                top_name = [name,]
                link_ncnn(layer, name, bottom_name, top_name)

        else:
            global num_bottleneck
            if isinstance(chimodule, Bottleneck):
                downsample = chimodule.downsample
                layer = convert['Split']('split')
                split_name0 = 'bottleneck' + str(num_bottleneck) + '_split0'
                split_name1 = 'bottleneck' + str(num_bottleneck) + '_split1'
                link_ncnn(layer, "split", top_name, [split_name0, split_name1])
                top_name = [split_name1,]
                #bottle_bottomname = top_name
                for comname, bottlecom in chimodule.named_children():
                    print(type(chimodule), bottlecom)
                    if 'relu' in comname:
                        if downsample is not None:
                            downsa_topname = [split_name0,]
                            for downcom in downsample.children():
                                bottom_name = downsa_topname
                                layer_type_name = downcom._get_name()
                                layer = convert[layer_type_name](downcom)
                                if layer_type_name == 'BatchNorm2d':
                                    name = ["downsample" + str(num_bottleneck) + downcom._get_name(),]
                                    downsa_topname = ["downsample" + str(num_bottleneck) + downcom._get_name() + 'scale',]
                                    link_ncnn(layer[0], name[0], bottom_name, name)
                                    link_ncnn(layer[1], downsa_topname[0], name, downsa_topname)
                                else:
                                    downsa_topname = ["downsample" + str(num_bottleneck) + downcom._get_name(),]
                                    link_ncnn(layer, downsa_topname[0], bottom_name, downsa_topname)
                            bottom_name =  top_name 
                            bottom_name.extend(downsa_topname)
                            top_name = ["eltwise" + str(num_bottleneck),]
                            layer = convert['Add']('none')
                            link_ncnn(layer, top_name[0], bottom_name, top_name)
                        else:
                            bottom_name =  top_name 
                            bottom_name.extend([split_name0,])
                            top_name = ["eltwise" + str(num_bottleneck),]
                            layer = convert['Add']('none')
                            link_ncnn(layer, top_name[0], bottom_name, top_name)
                        bottom_name = top_name
                        top_name = [str(num_bottleneck) + '_relu_bottleout']
                        layer = convert[bottlecom._get_name()](bottlecom)
                        link_ncnn(layer, top_name[0], bottom_name, top_name)
                    elif  'downsample' in comname:
                        continue
                    else:
                        bottom_name = top_name

                        layer_type_name = bottlecom._get_name()
                        layer = convert[layer_type_name](bottlecom)
                        if layer_type_name == 'BatchNorm2d':
                            name = ['bottleneck' + str(num_bottleneck)  + comname,]
                            top_name = ['bottleneck' + str(num_bottleneck)  + comname + 'scale',]
                            link_ncnn(layer[0], name[0], bottom_name, name)
                            link_ncnn(layer[1], top_name[0], name, top_name)
                            if 'bn3' not in comname:
                                bottom_name = top_name
                                top_name = [str(num_bottleneck) + comname + '_relu']
                                layer = convert['ReLU']('relu')
                                link_ncnn(layer, top_name[0], bottom_name, top_name)
                        else:
                            top_name = ['bottleneck' + str(num_bottleneck)  + comname,]
                            link_ncnn(layer, top_name[0], bottom_name, top_name)
                num_bottleneck = num_bottleneck + 1
                print(num_bottleneck)
               # print(chimodule, type(chimodule), chimodule.downsample)
            else:
                convert_ncnn(chimodule)

def get_model(models):
    global ncnn_net,ncnn_weights,layer_type_count,top_name,bottom_name,blob_set,num_bottleneck
    num_bottleneck = 0
    blob_set = set()
    top_name = []
    bottom_name = []
    layer_type_count = dict()
    ncnn_net = []
    ncnn_weights = []

    for model in models:
        convert_ncnn(model)

    text_net = '\n'.join(ncnn_net)
    """ Add layer number and blob number """
    text_net = ('%d %d\n' % (len(ncnn_net), len(blob_set))) + text_net
    """ Add ncnn magic number """
    text_net = '7767517\n' + text_net

    return text_net, ncnn_weights
