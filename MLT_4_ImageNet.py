# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:50:17 2020

@author: Luther Ollier


_________________________________Machine learning tool________________________________________________________
____________________________________part 4_________________________________________________________________
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

print("Pytorch version:{}".format(torch.__version__))

#useful links
#https://pytorch.org/docs/stable/_modules/torchvision/datasets/imagenet.html#ImageNet
#https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet

#%%
"""
#_________________________________Dataloader for ImageNet______________________________#
#                                                                                      #  
# The source code of the class ImageNet from pytorch says                              #
#                                                                                      #
# The dataset is no longer publicly accessible.                                        # 
# You need to download the archives externally and place them in the root directory.   #                                        
#                                                                                      #
#  I couldn't download the image because it takes 5 days to check a new account on     #
#  the ImageNet website so i just implement the dataloader with the class ImageNet     #
#  from pytorch                                                                        #
#                                                                                      #                                                                         
#______________________________________________________________________________________#
"""

root_dir="path_to_ImageNet_directory_file"

data=ImageNet(root=root_dir)
data_loader=DataLoader(data, batch_size=1)
