from convert_ncnn import *
from convert_caffe import *
import argparse
import torch
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CONVERT_IMAGE_CAPTION_MODEL')
    parser.add_argument('--model', '-m', default='BEST_checkpoint_.pth.tar', help='path to model')
    parser.add_argument('--dst', '-d', default='ncnn', help='the framework to convert(support ncnn,caffe)')
    args = parser.parse_args()


    
    #original encoder structure
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    #remove adaptivepool because caffe(nnie) don't support 
    '''
    modules = list(encoder.children())[:-1]
    encoder = nn.Sequential(*modules)
    '''

#   change decoder structure to get right order weights for ncnn.
    from models import DecoderWithAttention as Decoder
    decoder = Decoder(512, 512, 512, 11676)
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    decoder_dict = {}
    for name, parma in checkpoint['decoder'].state_dict().items():
            decoder_dict[name] = parma
    decoder.load_state_dict(decoder_dict)
    decoder.eval()

#   original decoder model, it also work, but the graph order is not we want.
    '''
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    '''

    if dst == 'ncnn':
        models = [decoder]
        text_net, binary_weights = get_model(models)
    #    print(text_net)
        NetName = 'imagecaption'#str(pytorch_net.__class__.__name__)
        ModelDir = './ncnn_model'
        import numpy as np
        with open(ModelDir  + '/' + NetName + '_decoder.param', 'w') as f:
            f.write(text_net)
        with open(ModelDir  + '/' + NetName + '_decoder_int8.bin', 'w') as f:
            for weights in binary_weights:
                for blob in weights:
                    blob_32f = blob.flatten().astype(np.int8)
                    blob_32f.tofile(f)
        print('Converting Done.')

    elif dst == 'caffe':
        text_net, binary_weights = get_caffemodel(models)

        ModelDir = './caffe_model'
        NetName = str(models.__class__.__name__)
        import numpy as np
        import google.protobuf.text_format
        with open(ModelDir + '/' + NetName + '.prototxt', 'w') as f:
            f.write(google.protobuf.text_format.MessageToString(text_net))
        with open(ModelDir + '/' + NetName + '.caffemodel', 'wb') as f:
            modelstring = binary_weights.SerializePartialToString()
            f.write(modelstring)
        print('Converting Done.')
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--model', '-m', default='BEST_checkpoint_.pth.tar', help='path to model')
    args = parser.parse_args()

    # Load model
#    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
#    decoder = checkpoint['decoder']
#    decoder = decoder.to(device)
#    decoder.eval()
#    encoder = checkpoint['encoder']
#    encoder = encoder.to(device)
#    encoder.eval()

    model_path = './BEST_checkpoint_.pth.tar'
    from models import DecoderWithAttention as Decoder
    decoder = Decoder(512, 512, 512, 11676)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    decoder_dict = {}
    for name, parma in checkpoint['decoder'].state_dict().items():
            decoder_dict[name] = parma
#           encoder_dict[name] = parma
#            print(name)

    decoder.load_state_dict(decoder_dict)
    decoder.eval()

#    a = torch.rand(5, 14, 14, 2048)
#    b = torch.randint(0, 11676, (5,50))
#    c = torch.randint(0,50,(5,1))
#    decoder = decoder.cpu()
#    output_var = decoder(a, b, c)#.permute(0, 2, 3, 1) 
#    pytorch_output = output_var.data.cpu().numpy()

#    modules = list(encoder.children())[:-1]
#    print(modules)
#    models = nn.Sequential(*modules)

if dst == 'ncnn':
    models = [encoder]
    text_net, binary_weights = get_model(models)
#    print(text_net)
    NetName = 'imagecaption'#str(pytorch_net.__class__.__name__)
    ModelDir = './wintest/'
    import numpy as np
    with open(ModelDir + NetName + '/' + NetName + 'decoder.param', 'w') as f:
        f.write(text_net)
    with open(ModelDir + NetName + '/' + NetName + 'decoder.bin', 'w') as f:
        for weights in binary_weights:
            for blob in weights:
                blob_32f = blob.flatten().astype(np.float32)
                blob_32f.tofile(f)
    print('Converting Done.'
elif dst == 'caffe':
    text_net, binary_weights = get_caffemodel(models)

    ModelDir = './caffe_model'
    NetName = str(models.__class__.__name__)
    import numpy as np
    import google.protobuf.text_format
    with open(ModelDir + '/' + NetName+"test" + '.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(text_net))
    with open(ModelDir + '/' + NetName + '.caffemodel', 'wb') as f:
        modelstring = binary_weights.SerializePartialToString()
        f.write(modelstring)
    print('Converting Done.')
'''
