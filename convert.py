import numpy as np

class LayerParameter_ncnn(object):

    def __init__(self):
        self.type = ''
        self.param = []
        self.weights = []

def spatial_convolution(chimodule):
#    print(type(chimodule.state_dict()['weight']))
    weight = chimodule.state_dict()['weight'].cpu().numpy() 
#    print(type(weight))
    try:
        bias = chimodule.state_dict()['bias'].cpu().numpy()
    except:
        bias = []
    
    (out_channels, in_channels, kh, kw) = weight.shape
    padH, padW = chimodule.padding
    strH, strW = chimodule.stride
    dilation = chimodule.dilation[0]
    groups = chimodule.groups

    assert kh == kw, [kh, kw]
    assert strH == strW, [strH, strW]
    assert padH == padW, [padH, padW]
    
#    print(chimodule._get_name(), chimodule, out_channels)
    layer = LayerParameter_ncnn()
    layer.type = 'Convolution'
    layer.param.append('%d' % out_channels)
    layer.param.append('%d' % kh)
    layer.param.append('%d' % dilation)
    layer.param.append('%d' % strH)
    layer.param.append('%d' % padH)


    if len(bias):
        layer.param.append('%d' % True)
        layer.param.append('%d' % weight.size)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)
        layer.weights.append(bias)
    else:
        layer.param.append('%d' % False)
        layer.param.append('%d' % weight.size)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)

    return layer

def ReLU(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'ReLU'
    layer.param.append('%f' % 0.0)
    return layer

def MaxPooling(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'Pooling'
    layer.param.append('%d' % 0)
    padH = chimodule.padding
    strH = chimodule.stride
    kH = chimodule.kernel_size 
    layer.param.append('%d' % kH)
    layer.param.append('%d' % strH)
    layer.param.append('%d' % padH)
    layer.param.append('%d' % 0)    
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

def inner_product(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'InnerProduct'   
    weight = chimodule.state_dict()['weight'].cpu().numpy() 
#    print(type(weight))
    try:
        bias = chimodule.state_dict()['bias'].cpu().numpy()
    except:
        bias = []

    num_output = chimodule.out_features
    layer.param.append('%d' % num_output)

    if len(bias):
#        print("***************************inner_product_bias")
        layer.param.append('%d' % True)
        layer.param.append('%d' % weight.size)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)
        layer.weights.append(bias)
    else:
        layer.param.append('%d' % False)
        layer.param.append('%d' % weight.size)
#        layer.weights.append(np.array([0.]))
        layer.weights.append(weight)
    
    return layer
        
def AdaptiveAvgPooling(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'AdaptiveAvgPooling'
    layer.param.append('%d' % chimodule.output_size[0])
    layer.param.append('%d' % chimodule.output_size[1])
#    CopyPoolingParameter(pytorch_layer, layer)
    return layer

def batchnorm(chimodule):
    layer_bn = LayerParameter_ncnn()
    layer_bn.type = 'BatchNorm'

    layer_bn.param.append('%d' % chimodule.running_mean.cpu().numpy().size)

#    layer_bn.weights.append(np.ones(chimodule.running_mean.cpu().numpy().shape))

#    layer_bn.weights.append(np.zeros(chimodule.running_mean.cpu().numpy().shape))

#    layer_scale = LayerParameter_ncnn()
#    layer_scale.type = 'Scale'

    weight = chimodule.state_dict()['weight'].cpu().numpy() 
#    print(type(weight))
    try:
        bias = chimodule.state_dict()['bias'].cpu().numpy()
    except:
        bias = []
    
    if len(bias):
#        print("***************************batchnorm_bias")
#        layer_scale.param.append('%d' % weight.size)
#        layer_scale.param.append('%d' % True)
        layer_bn.weights.append(weight)
        layer_bn.weights.append(bias)

        layer_bn.weights.append(chimodule.running_mean.cpu().numpy())
        """ Add eps by hand for running_var in ncnn """
        running_var = chimodule.running_var.cpu().numpy()
        running_var = running_var + chimodule.eps
        layer_bn.weights.append(running_var)
    else:
        layer_bn.weights.append(weight)
        layer_bn.weights.append(chimodule.running_mean.cpu().numpy())
        """ Add eps by hand for running_var in ncnn """
        running_var = chimodule.running_var.cpu().numpy()
        running_var = running_var + chimodule.eps
        layer_bn.weights.append(running_var)
        layer_bn.weights.append(np.zeros(chimodule.running_mean.cpu().numpy().shape))
#        layer_scale.param.append('%d' % weight.size)
#        layer_scale.param.append('%d' % False)
#        layer_scale.weights.append(weight)

    return layer_bn#[layer_bn, layer_scale]

def dropout(chimodule):
    layer = LayerParameter_ncnn()
    dropout_ratio = 0.5#float(pytorch_layer.p)
    layer.type = 'Dropout'
    if abs(dropout_ratio - 0.5) < 1e-3:
        pass
    else:
        scale = 1.0 - dropout_ratio
        layer.param.append('%f' % scale)
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

def softmax(chimodule):
    layer = LayerParameter_ncnn()
    layer.type = 'Softmax'
    """ TODO: axis """
    layer.param.append('%d' % 0)

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
    pp.append('%-16s %d %d' % (name, 1, 1))
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
    global ncnn_net,ncnn_weights,layer_type_count,top_name,bottom_name
    for chinam, chimodule in inModule.named_children():
        if len(chimodule._modules) == 0:
#            print(chimodule._get_name(), 'func->', chimodule)
#            print('type', type(chimodule))
#            if 'Conv' in chimodule._get_name():
            convert = look_for_convertlayer()
            layer_type_name = chimodule._get_name()
            if layer_type_name in layer_type_count:
                layer_type_count[layer_type_name] += 1
            else:
                layer_type_count[layer_type_name] = 1
            name = layer_type_name + '_' + str(layer_type_count[layer_type_name])
            print(layer_type_name)
#            print(type(chimodule))
            layer = convert[layer_type_name](chimodule)
#            if layer_type_name == 'BatchNorm2d':
#                scale_name = name + '_' + 'scale'
#                bottom_name = top_name
#                top_name = scale_name
#                link_ncnn(layer[0], name, [bottom_name,], [name,])
#                link_ncnn(layer[1], scale_name, [name,], [scale_name])
          #  else:
            bottom_name = top_name
            top_name = name
            link_ncnn(layer, name, [bottom_name,], [top_name,])

        else:
            convert_ncnn(chimodule)

def get_model(models):
    global ncnn_net,ncnn_weights,layer_type_count,top_name,bottom_name,blob_set
    blob_set = set()
    top_name = ""
    bottom_name = ""
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