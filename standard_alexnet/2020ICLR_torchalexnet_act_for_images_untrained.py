import argparse
import os
import scipy.io
import pickle
import time
from collections import OrderedDict

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import torchvision.models
from torch.utils import model_zoo

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def compute_features(dataloader, model, N):
    model.eval()
    act = {}

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(output.cpu().numpy())

    for m in model.features.modules():
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)

    for m in model.classifier.modules():
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)

    for i, input_tensor in enumerate(dataloader):
        with torch.no_grad():
            input_var, label = input_tensor[0].cuda(),input_tensor[2]
            _model_feats = []
            aux = model(input_var).data.cpu().numpy()
            act[label[0]] = _model_feats

    return act



def get_activations(offset):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]
#        dataset = datasets.ImageFolder(offset, transform=transforms.Compose(tra))
    dataset = ImageFolderWithPaths(offset, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            num_workers=0,
                                            pin_memory=True,
                                            shuffle = False)
    features = compute_features(dataloader, model, len(dataset))
    return features


if __name__ == '__main__':

    image_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/92_Image_Set/92images' 
    model = torchvision.models.alexnet(pretrained = False)
    model.cuda()
    act = get_activations(image_pth)
    outpth = '/home/CUSACKLAB/annatruzzi/cichy2016/niko92_activations_untrained_torchalexnet.pickle'
    with open(outpth, 'wb') as handle:
        pickle.dump(act, handle)
