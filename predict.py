import os, sys
import argparse
import numpy as np
import torch
import torchvision.transforms as t

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from alexnet import KitModel as AlexNet
from vgg19 import KitModel as VGG19


class ImageListDataset (Dataset):

    def __init__(self, list_filename, root=None, transform=None):
        super(ImageListDataset).__init__()
    
        with open(list_filename, 'r') as list_file:
            self.list = list(map(str.rstrip, list_file))
        
        self.root = root
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.list[index]
        if self.root:
            path = os.path.join(self.root, path)
            
        x = default_loader(path)
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.list)


def main(args):

    transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Lambda(lambda x: x[[2,1,0], ...] * 255),  # RGB -> BGR and [0,1] -> [0,255]
        t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1,1,1]),  # mean subtraction
    ])

    data = ImageListDataset(args.image_list, root=args.root, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    model = AlexNet if 'hybrid' in args.model else VGG19
    model = model('converted-models/{}.pth'.format(args.model)).to('cuda')
    model.eval()
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            p = model(x.to('cuda')).cpu().numpy()  # order is (NEG, NEU, POS)
            np.savetxt(sys.stdout.buffer, p, delimiter=',')

    
if __name__ == '__main__':
    models = ('hybrid_finetuned_fc6+',
          'hybrid_finetuned_all',
          'vgg19_finetuned_fc6+',
          'vgg19_finetuned_all')

    parser = argparse.ArgumentParser(description='Predict Visual Sentiment')
    parser.add_argument('image_list', type=str, help='Image list (txt, one path per line)')
    parser.add_argument('-r', '--root', default=None, help='Root path to prepend to image list')
    parser.add_argument('-m', '--model', type=str, choices=models, default='vgg19_finetuned_all', help='Pretrained model')
    parser.add_argument('-b', '--batch-size', type=int, default=48, help='Batch size')
    args = parser.parse_args()
    main(args)
