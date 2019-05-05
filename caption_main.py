from convert import *
import argparse
import torch

device = "cuda"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--model', '-m', default='BEST_checkpoint_.pth.tar', help='path to model')
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    models = [encoder, decoder]
    text_net, binary_weights = get_model(models)
#    print(text_net)
    NetName = 'imagecaption_encoder'#str(pytorch_net.__class__.__name__)
    ModelDir = './wintest/'
    import numpy as np
    with open(ModelDir + NetName + '/' + NetName + '.param', 'w') as f:
        f.write(text_net)
    with open(ModelDir + NetName + '/' + NetName + '.bin', 'w') as f:
        for weights in binary_weights:
            for blob in weights:
                blob_32f = blob.flatten().astype(np.float32)
                blob_32f.tofile(f)
    print('Converting Done.')
