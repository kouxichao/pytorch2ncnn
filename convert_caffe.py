"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import math
import numpy as np
from caffe.proto import caffe_pb2 as pb2

def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob


def CopyTuple(param):
    if isinstance(param, tuple):
        return param
    elif isinstance(param, int):
        return param, param
    else:
        assert type(param)


def ty(caffe_type):
    def f(_):
        layer = pb2.LayerParameter()
        layer.type = caffe_type
        return layer
    return f


def data(inputs):
    layer = pb2.LayerParameter()
    layer.type = 'Input'
    input_shape = pb2.BlobShape()
    input_shape.dim.extend(inputs.data.numpy().shape)
    layer.input_param.shape.extend([input_shape])
    return layer


def Slice(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Slice"

    layer.slice_param.axis = pytorch_layer.dim
    return layer


def inner_product(pytorch_layer):
#    global writed_list
    layer = pb2.LayerParameter()
    layer.type = "InnerProduct"
    
#    blobs = pytorch_layer.next_functions[2][0].next_functions[0][0]

#    if blobs in writed_list:
#       return
#    else:
#       writed_list.append(blobs)

    blobs_weight = pytorch_layer.state_dict()['weight'].cpu().numpy() 
    num_output = pytorch_layer.out_features
    layer.inner_product_param.num_output = num_output

    try:
        bias = pytorch_layer.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    if len(bias):
        layer.inner_product_param.bias_term = True
        layer.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer.inner_product_param.bias_term = False
        layer.blobs.extend([as_blob(blobs_weight)])

    return layer


def concat(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Concat"
    layer.concat_param.axis = int(pytorch_layer.dim)
    return layer


def flatten(pytorch_layer):
    """ Only support flatten view """
    total = 1
    for dim in pytorch_layer.old_size:
        total *= dim
#    assert ((pytorch_layer.new_sizes[1] == total) or (pytorch_layer.new_sizes[1] == -1))

    layer = pb2.LayerParameter()
    layer.type = "Flatten"
    return layer


def spatial_convolution(pytorch_layer):
    layer = pb2.LayerParameter()
   # print(pytorch_layer.metadata)	
#    print(dir(pytorch_layer),"s")
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

    if pytorch_layer.transposed:
        layer.type = "Deconvolution"
        layer.convolution_param.num_output = nInputPlane
    else:
        layer.type = "Convolution"
        layer.convolution_param.num_output = nOutputPlane

    if kH == kW:
        layer.convolution_param.kernel_size.extend([kH])
    else:
        layer.convolution_param.kernel_h = kH
        layer.convolution_param.kernel_w = kW
    if dH == dW:
        layer.convolution_param.stride.extend([dH])
    else:
        layer.convolution_param.stride_h = dH
        layer.convolution_param.stride_w = dW
    if padH == padW:
        layer.convolution_param.pad.extend([padH])
    else:
        layer.convolution_param.pad_h = padH
        layer.convolution_param.pad_w = padW
    layer.convolution_param.dilation.extend([dilation])
    layer.convolution_param.group = pytorch_layer.groups

    if len(bias):
        layer.convolution_param.bias_term = True
        layer.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer.convolution_param.bias_term = False
        layer.blobs.extend([as_blob(blobs_weight)])

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
    layer = pb2.LayerParameter()
    layer.type = "Deconvolution"

    assert pytorch_layer.scale_factor[0] == pytorch_layer.scale_factor[1]
    factor = int(pytorch_layer.scale_factor[0])
    c = int(pytorch_layer.input_size[1])
    k = 2 * factor - factor % 2

    layer.convolution_param.num_output = c
    layer.convolution_param.kernel_size.extend([k])
    layer.convolution_param.stride.extend([factor])
    layer.convolution_param.pad.extend([int(math.ceil((factor - 1) / 2.))])
    layer.convolution_param.group = c
    layer.convolution_param.weight_filler.type = 'bilinear'
    layer.convolution_param.bias_term = False

    learning_param = pb2.ParamSpec()
    learning_param.lr_mult = 0
    learning_param.decay_mult = 0
    layer.param.extend([learning_param])

    """ Init weight blob of filter kernel """
    blobs_weight = FillBilinear(c, k)
    layer.blobs.extend([as_blob(blobs_weight)])

    return layer


def CopyPoolingParameter(pytorch_layer, layer):

    kH, kW = CopyTuple(pytorch_layer.kernel_size)
    dH, dW = CopyTuple(pytorch_layer.stride)
    padH, padW = CopyTuple(pytorch_layer.padding)

    if kH == kW:
        layer.pooling_param.kernel_size = kH
    else:
        layer.pooling_param.kernel_h = kH
        layer.pooling_param.kernel_w = kW
    if dH == dW:
        layer.pooling_param.stride = dH
    else:
        layer.pooling_param.stride_h = dH
        layer.pooling_param.stride_w = dW
    if padH == padW:
        layer.pooling_param.pad = padH
    else:
        layer.pooling_param.pad_h = padH
        layer.pooling_param.pad_w = padW

    if pytorch_layer.ceil_mode is True:
        return

    if pytorch_layer.ceil_mode is False:
        if padH == padW:
            if dH > 1 and padH > 0:
                layer.pooling_param.pad = padH - 1
        else:
            if dH > 1 and padH > 0:
                layer.pooling_param.pad_h = padH - 1
            if dW > 1 and padW > 0:
                layer.pooling_param.pad_w = padW - 1


def MaxPooling(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    layer.pooling_param.pool = pb2.PoolingParameter.MAX
    CopyPoolingParameter(pytorch_layer, layer)
    return layer


def AvgPooling(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    layer.pooling_param.pool = pb2.PoolingParameter.AVE
    CopyPoolingParameter(pytorch_layer, layer)
    return layer

def dropout(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Dropout"
    layer.dropout_param.dropout_ratio = float(pytorch_layer.p)
    train_only = pb2.NetStateRule()
    train_only.phase = pb2.TEST
    layer.exclude.extend([train_only])
    return layer


def elu(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "ELU"
    layer.elu_param.alpha = pytorch_layer.additional_args[0]
    return layer


def leaky_ReLU(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "ReLU"
    layer.relu_param.negative_slope = float(pytorch_layer.additional_args[0])
    return layer


def PReLU(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "PReLU"
    num_parameters = int(pytorch_layer.num_parameters)
    layer.prelu_param.channel_shared = True if num_parameters == 1 else False

    blobs_weight = pytorch_layer.next_functions[1][0].variable.data.numpy()
    layer.blobs.extend([as_blob(blobs_weight)])
    return layer


def MulConst(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Power"
    layer.power_param.power = 1
    layer.power_param.scale = float(pytorch_layer.constant)
    layer.power_param.shift = 0
    return layer


def AddConst(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Power"
    layer.power_param.power = 1
    layer.power_param.scale = 1
    """ Constant to add should be filled by hand, since not visible in autograd """
    layer.power_param.shift = float('inf')
    return layer


def softmax(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = 'Softmax'
    return layer


def eltwise(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    return layer

def Reshape(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Reshape"
    return layer

def Per(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "transpose"


def eltwise_max(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    layer.eltwise_param.operation = 2
    return layer


def batchnorm(pytorch_layer):
    layer_bn = pb2.LayerParameter()
    layer_bn.type = "BatchNorm"

    layer_bn.batch_norm_param.use_global_stats = 1
    layer_bn.batch_norm_param.eps = pytorch_layer.eps
    layer_bn.blobs.extend([
        as_blob(pytorch_layer.running_mean.numpy()),
        as_blob(pytorch_layer.running_var.numpy()),
        as_blob(np.array([1.]))
    ])

    layer_scale = pb2.LayerParameter()
    layer_scale.type = "Scale"

    blobs_weight = pytorch_layer.state_dict()['weight'].cpu().numpy()
    try:
        bias = pytorch_layer.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    if len(bias):
        layer_scale.scale_param.bias_term = True
        layer_scale.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer_scale.scale_param.bias_term = False
        layer_scale.blobs.extend([as_blob(blobs_weight)])

    return [layer_bn, layer_scale]


def build_converter():
    return {
        'data': data,
        'Linear': inner_product,
        'ReLU':ty('ReLU'),
        'Threshold': ty('ReLU'),
        'Conv2d': spatial_convolution,
	    #'ThnnConv2D': spatial_convolution,
        'MaxPool2d': MaxPooling,
        'AvgPool2d': AvgPooling,
#        'AdaptiveAvgPool2d': AdaptiveAvgPooling,
        'Add': eltwise,
        'Cmax': eltwise_max,
        'BatchNorm2d': batchnorm,
        'Concat': concat,
        'Dropout': dropout,
        'UpsamplingBilinear2d': UpsampleBilinear,
        'MulConstant': MulConst,
        'AddConstant': AddConst,
        'Softmax': softmax,
        'Sigmoid': ty('Sigmoid'),
        'Tanh': ty('TanH'),
        'ELU': elu,
        'LeakyReLU': leaky_ReLU,
        'PReLU': PReLU,
        'Chunk': Slice,
        'View': flatten,
        'Squeeze': Reshape,
        'Permute': Per,
    }

def link_caffe(layer, name, bottom, top):
    layer.name = name
    for b in bottom:
        layer.bottom.append(b)
    for t in top:
        layer.top.append(t)

    caffe_net.append(layer)

def convert_caffe(inModule):
    """
    """
    from torchvision.models.resnet import Bottleneck
    global layer_type_count,top_name,bottom_name
    for chinam, chimodule in inModule.named_children():
        convert = build_converter()
        if len(chimodule._modules) == 0:
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
                link_caffe(layer[0], name, bottom_name, [name,])
                link_caffe(layer[1], scale_name, [name,], top_name)
            else:
                bottom_name = top_name
                top_name = [name,]
                link_caffe(layer, name, bottom_name, top_name)

        else:
            global num_bottleneck
            if isinstance(chimodule, Bottleneck):
                downsample = chimodule.downsample
                bottle_bottomname = top_name
                for comname, bottlecom in chimodule.named_children():
                    print(type(chimodule), bottlecom)
                    if 'relu' in comname:
                        if downsample is not None:
                            downsa_topname = bottle_bottomname
                            for downcom in downsample.children():
                                bottom_name = downsa_topname
                                layer_type_name = downcom._get_name()
                                layer = convert[layer_type_name](downcom)
                                if layer_type_name == 'BatchNorm2d':
                                    name = ["downsample" + str(num_bottleneck) + downcom._get_name(),]
                                    downsa_topname = ["downsample" + str(num_bottleneck) + downcom._get_name() + 'scale',]
                                    link_caffe(layer[0], name[0], bottom_name, name)
                                    link_caffe(layer[1], downsa_topname[0], name, downsa_topname)
                                else:
                                    downsa_topname = ["downsample" + str(num_bottleneck) + downcom._get_name(),]
                                    link_caffe(layer, downsa_topname[0], bottom_name, downsa_topname)
                            bottom_name =  top_name 
                            bottom_name.extend(downsa_topname)
                            top_name = ["eltwise" + str(num_bottleneck),]
                            layer = convert['Add']('none')
                            link_caffe(layer, top_name[0], bottom_name, top_name)
                        else:
                            bottom_name =  top_name 
                            bottom_name.extend(bottle_bottomname)
                            top_name = ["eltwise" + str(num_bottleneck),]
                            layer = convert['Add']('none')
                            link_caffe(layer, top_name[0], bottom_name, top_name)
                        bottom_name = top_name
                        top_name = [str(num_bottleneck) + '_relu_bottleout']
                        layer = convert[bottlecom._get_name()](bottlecom)
                        link_caffe(layer, top_name[0], bottom_name, top_name)
                    elif  'downsample' in comname:
                        continue
                    else:
                        bottom_name = top_name

                        layer_type_name = bottlecom._get_name()
                        layer = convert[layer_type_name](bottlecom)
                        if layer_type_name == 'BatchNorm2d':
                            name = ['bottleneck' + str(num_bottleneck)  + comname,]
                            top_name = ['bottleneck' + str(num_bottleneck)  + comname + 'scale',]
                            link_caffe(layer[0], name[0], bottom_name, name)
                            link_caffe(layer[1], top_name[0], name, top_name)
                            if 'bn3' not in comname:
                                bottom_name = top_name
                                top_name = [str(num_bottleneck) + comname + '_relu']
                                layer = convert['ReLU']('relu')
                                link_caffe(layer, top_name[0], bottom_name, top_name)
                        else:
                            top_name = ['bottleneck' + str(num_bottleneck)  + comname,]
                            link_caffe(layer, top_name[0], bottom_name, top_name)
                num_bottleneck = num_bottleneck + 1
                print(num_bottleneck)
               # print(chimodule, type(chimodule), chimodule.downsample)
            else:
                convert_caffe(chimodule)

def get_caffemodel(models):
    import os
#    import caffe_pb2 as pb2    

    global caffe_net,layer_type_count,top_name,bottom_name, num_bottleneck
    num_bottleneck = 0
    top_name = []
    bottom_name = []
    layer_type_count = dict()
    caffe_net = []

    convert_caffe(models)
    
    """ Caffe input """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True

    """ Caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for layer in caffe_net:
        binary_weights.layer.extend([layer])

        layer_proto = pb2.LayerParameter()
        layer_proto.CopyFrom(layer)
        del layer_proto.blobs[:]
        text_net.layer.extend([layer_proto])

    print(binary_weights.ByteSize())
    return text_net, binary_weights


