import argparse
import caffe
import glob
import torch

import numpy as np
import torchvision.transforms.functional as F

from PIL import Image

from alexnet import KitModel as AlexNet
from vgg19 import KitModel as VGG19

if __name__ == '__main__':
    models = ('hybrid_finetuned_fc6+',
              'hybrid_finetuned_all',
              'vgg19_finetuned_fc6+',
              'vgg19_finetuned_all')

    parser = argparse.ArgumentParser(description='Check outputs of original and converted models')
    parser.add_argument('model', type=str, choices=models, help='model to test')
    parser.add_argument('-i', '--image', type=str, default='dummy-data/lenna.jpg', help='input image')
    args = parser.parse_args()

    model = AlexNet if 'hybrid' in args.model else VGG19
    
    converted_model_weights = 'converted-models/{}.pth'.format(args.model)
    converted_model = model(converted_model_weights)
    converted_model.eval()
    
    original_model_net = 'original-models/{}/deploy.prototxt'.format(args.model)
    original_model_weights = 'original-models/{}/snapshot_iter_*.caffemodel'.format(args.model)
    original_model_weights = glob.glob(original_model_weights)[0]
    original_model = caffe.Net(original_model_net, caffe.TEST, weights=original_model_weights)
    
    # image
    pil_image = Image.open(args.image).convert('RGB')
    image = F.to_tensor(F.resize(pil_image, (224, 224)))  # resize to 224
    image = image[[2,1,0]] * 255 # RGB -> BGR (expected by caffe nets), [0,1] -> [0, 255]
    
    # mean
    mean_file = 'original-models/{}/mean.binaryproto'.format(args.model)
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open(mean_file, 'rb').read())
    mean_image = caffe.io.blobproto_to_array(blob).squeeze().astype(np.uint8)
    mean_pixel = torch.from_numpy(mean_image.mean(axis=(1,2), keepdims=True).astype(np.float32))
    
    print(mean_pixel)
    
    # input
    net_input = (image - mean_pixel).unsqueeze(0)
    
    # forward
    original_model.blobs['data'].data[...] = net_input
    original_output = original_model.forward()
    
    converted_output = converted_model(net_input)
    
    # outputs
    print(original_output)
    print(converted_output)
    
