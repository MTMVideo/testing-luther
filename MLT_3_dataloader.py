# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:04:34 2020

@author: Luther Ollier


_________________________________Machine learning tool________________________________________________________
____________________________________part 3_________________________________________________________________
"""

import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np 

print("Pytorch version:{}".format(torch.__version__))
print("Numpy version:{}".format(np.__version__))

#useful links
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

#%%
"""
#_________________________________A simple Dataloader for image________________________#
#                                                                                      #  
#                                                                                      #
# Overview :                                                                           #              
#                                                                                      #                                                            #
#                                                                                      #
# - Custom data set : inherit of the dataset class from pytorch                        #
#                                                                                      #
#                                                                                      #
# - resizer : transformation that we will apply to our dataset                         #
#                                                                                      #
#                                                                                      #
# - Creating the loader and load the images                                            #
#                                                                                      #
#______________________________________________________________________________________#
"""


class fixed_size_image_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.listdir(self.root_dir)[idx]
        sample = Image.open(os.path.join(self.root_dir,img_name))

        if self.transform:
            sample = self.transform(sample)

        return sample

def resize(image,size=(256,256)): 
    return image.resize(size)

tran = transforms.Compose([resize,transforms.ToTensor()])

path=r'.\image'

image_dataset = fixed_size_image_dataset(root_dir=path,
                                           transform=tran)

dataset_loader = DataLoader(image_dataset, shuffle=True)

for i,image in enumerate(dataset_loader):
    print(image.size())

    
