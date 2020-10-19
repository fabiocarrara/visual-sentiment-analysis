import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes', allow_pickle=True).item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1_1 = self.__conv(2, name='conv1_1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv1_2 = self.__conv(2, name='conv1_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_1 = self.__conv(2, name='conv2_1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_2 = self.__conv(2, name='conv2_2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_1 = self.__conv(2, name='conv3_1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_2 = self.__conv(2, name='conv3_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_3 = self.__conv(2, name='conv3_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_4 = self.__conv(2, name='conv3_4', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_1 = self.__conv(2, name='conv4_1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_2 = self.__conv(2, name='conv4_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_3 = self.__conv(2, name='conv4_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_4 = self.__conv(2, name='conv4_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_1 = self.__conv(2, name='conv5_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_2 = self.__conv(2, name='conv5_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_3 = self.__conv(2, name='conv5_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_4 = self.__conv(2, name='conv5_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc6_1 = self.__dense(name = 'fc6_1', in_features = 25088, out_features = 4096, bias = True)
        self.fc7_1 = self.__dense(name = 'fc7_1', in_features = 4096, out_features = 4096, bias = True)
        self.fc8_retrain_1 = self.__dense(name = 'fc8-retrain_1', in_features = 4096, out_features = 3, bias = True)

    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        relu1_1         = F.relu(conv1_1)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        relu2_1         = F.relu(conv2_1)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        relu3_1         = F.relu(conv3_1)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        relu3_2         = F.relu(conv3_2)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        relu3_3         = F.relu(conv3_3)
        conv3_4_pad     = F.pad(relu3_3, (1, 1, 1, 1))
        conv3_4         = self.conv3_4(conv3_4_pad)
        relu3_4         = F.relu(conv3_4)
        pool3_pad       = F.pad(relu3_4, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        relu4_1         = F.relu(conv4_1)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        relu4_2         = F.relu(conv4_2)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        relu4_3         = F.relu(conv4_3)
        conv4_4_pad     = F.pad(relu4_3, (1, 1, 1, 1))
        conv4_4         = self.conv4_4(conv4_4_pad)
        relu4_4         = F.relu(conv4_4)
        pool4_pad       = F.pad(relu4_4, (0, 1, 0, 1), value=float('-inf'))
        pool4           = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        relu5_1         = F.relu(conv5_1)
        conv5_2_pad     = F.pad(relu5_1, (1, 1, 1, 1))
        conv5_2         = self.conv5_2(conv5_2_pad)
        relu5_2         = F.relu(conv5_2)
        conv5_3_pad     = F.pad(relu5_2, (1, 1, 1, 1))
        conv5_3         = self.conv5_3(conv5_3_pad)
        relu5_3         = F.relu(conv5_3)
        conv5_4_pad     = F.pad(relu5_3, (1, 1, 1, 1))
        conv5_4         = self.conv5_4(conv5_4_pad)
        relu5_4         = F.relu(conv5_4)
        pool5_pad       = F.pad(relu5_4, (0, 1, 0, 1), value=float('-inf'))
        pool5           = F.max_pool2d(pool5_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        fc6_0           = pool5.view(pool5.size(0), -1)
        fc6_1           = self.fc6_1(fc6_0)
        relu6           = F.relu(fc6_1)
        drop6           = F.dropout(input = relu6, p = 0.5, training = self.training, inplace = True)
        fc7_0           = drop6.view(drop6.size(0), -1)
        fc7_1           = self.fc7_1(fc7_0)
        relu7           = F.relu(fc7_1)
        drop7           = F.dropout(input = relu7, p = 0.5, training = self.training, inplace = True)
        fc8_retrain_0   = drop7.view(drop7.size(0), -1)
        fc8_retrain_1   = self.fc8_retrain_1(fc8_retrain_0)
        softmax         = F.softmax(fc8_retrain_1)
        return softmax


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

if __name__ == '__main__':
    converted_model = KitModel('vgg19_finetuned_all.pth')
    converted_model.eval()
    
    import caffe
    original_model = caffe.Net('deploy.prototxt', caffe.TEST, weights='snapshot_iter_74880.caffemodel')
    
    from PIL import Image
    import scipy
    import torchvision.transforms.functional as tf
    
    # image
    pil_image = Image.open('../../dummy-data/neutral.jpeg').convert('RGB')
    #image = np.array(pil_image).astype(np.float32)
    #image = scipy.misc.imresize(image, (224, 224), 'bilinear')
    #image = image.transpose((2, 0, 1))  # HWC to CHW
    #image = image[[2,1,0]]  # RGB to BGR
    image = tf.to_tensor(tf.resize(pil_image, (224, 224)))  # resize to 224
    image = image[[2,1,0]] * 255 # RGB -> BGR (expected by caffe nets), [0,1] -> [0, 255]
    
    # mean
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open('mean.binaryproto', 'rb').read())
    mean_image = caffe.io.blobproto_to_array(blob).squeeze().astype(np.uint8)
    mean_pixel = torch.from_numpy(mean_image.mean(axis=(1,2), keepdims=True).astype(np.float32))
    
    print(mean_pixel)
    
    # pil_mean_image = tf.to_pil_image(torch.from_numpy(mean_image))
    # mean_image = tf.to_tensor(tf.resize(pil_mean_image, 224))
    
    # input
    net_input = (image - mean_pixel).unsqueeze(0)
    print(net_input.mean())
    
    # forward
    original_model.blobs['data'].data[...] = net_input
    original_output = original_model.forward()
    
    converted_output = converted_model(net_input)
    
    # outputs
    print(original_output)
    print(converted_output)
    
