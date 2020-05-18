# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:44:36 2020

@author: Luther Ollier

_________________________________Auto-encoder on audio files_________________________________________________________
______________________________________Auto-encoder _________________________________________________

"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os

import librosa
import librosa.display


for i,j in zip([torch,np,librosa],["torch","numpy","librosa"]):
    print("{} version :".format(j) + i.__version__)

#%%
"""
#_________________________________Autoencoder on AudioFile_____________________________#
#                                                                                      #  
# The data comprises 23,682 audio files in wav format.                                 #
# Each file is about 1 second in length and contains one word from:                    #
#                                                                                      #
#               up, down, left, right, yes, no, on, off, stop, go.                     #
#                                                                                      #
# The audio files are organized into folders based on the word they contain.           #
#                                                                                      #
# Overview :                                                                           #              
#                                                                                      #                                                        
#                                                                                      #
# - Data preprocessing : building a dataloader that will transform and pad our audio   #
#                        files.                                                        #
#                        Because of limited calculus capacity and limited time,        #
#                        only 4 words will be use and the number of file for each      #
#                        will be reduce.                                               #
#                                                                                      #                         
# - Model definition : Autoencoder with encode layers and decode layers                #
#                                                                                      #
#                                                                                      #
# - Training :  train and save the model                                               #
#                                                                                      #
#                                                                                      #
#______________________________________________________________________________________#
"""


#%%Data preprocessing


def pad(mfcc_matrix):
    """
    A few of the audio files have differents size after the mfcc transformation,
    this function pad the matrix to the shape (40,44)
    """
    width, height = mfcc_matrix.shape[0], mfcc_matrix.shape[1]
    if width % 2 == 0:
        width_padding = ((40-width)//2, (40-width)//2)
    else:
        width_padding = ((40-width)//2 + 1, (40-width)//2)
    if height % 2 == 0:
        height_padding = ((44-height)//2, (44-height)//2)
    else:
        height_padding = ((44-height)//2 + 1, (44-height)//2)
    return np.pad(mfcc_matrix, (width_padding,height_padding), mode='edge')


def load(path):
    """
    Loading an audio file, apply the mfcc transformation and pad if necessary
    """
    data, sample_rate=librosa.load(path)
    mfcc=librosa.feature.mfcc(data,sample_rate, n_mfcc=40)
    if mfcc.shape != (40,44):
        mfcc = pad(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

class Custom_Audio_dataset(Dataset):

    def __init__(self, root_dir, word_list, transform=None, reduce=1.):
        """
        Args:
            root_dir (string): Directory with the folder of the audio files.
            word_list (list of string): the name of the differents folder containing the audio files
            transform (callable, optional): Optional transform to be applied
            on a sample.
            reduce (float): divide the total number of files used 
            """
        self.root_dir = root_dir
        self.word_list=word_list
        self.transform = transform
        self.reduce = reduce

    def __len__(self):
        total=0
        for word in self.word_list: # sum the lenght of each audio files to get the global lenght
            total+=int(len(os.listdir(os.path.join(self.root_dir,word+'/')))/self.reduce)
        return total

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        word_idx=0
        #as all our files are not in the same folder, browse the differents folder according to the index size
        while idx>=len(os.listdir(os.path.join(self.root_dir,self.word_list[word_idx]+'/')))/self.reduce:
            idx-=int(len(os.listdir(os.path.join(self.root_dir,self.word_list[word_idx]+'/')))/self.reduce)
            word_idx+=1
        audio_path = os.path.join(self.root_dir,self.word_list[word_idx]+'/')
        audio_path = os.path.join(audio_path,os.listdir(audio_path)[idx])
        label = self.word_list[word_idx]
        sample = load(audio_path) 
        if self.transform:
            sample = self.transform(sample)

        return sample,label
    
tran = transforms.Compose([transforms.ToTensor()])

path=r'./audio/'
word_list=["up", "down", "left", "right"]#, "yes", "no", "on", "off", "stop", "go"] 
#because of lack of time to compute all the calcul i reduced my dataset to four word.

audio_dataset = Custom_Audio_dataset(root_dir=path,
                                     word_list=word_list,
                                     transform=tran,
                                     reduce=3.)



#%%Model creation

class Autoencoder(nn.Module):
    def __init__(self, in_features):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Linear(in_features, 100)
        self.encoder2 = nn.Linear(100, 4)
        self.decoder1 = nn.Linear(4, 100)
        self.decoder2 = nn.Linear(100, in_features)
    def forward(self, x):
        x = F.relu(self.encoder1(x))
        code = F.relu(self.encoder2(x))
        x = F.relu(self.decoder1(code))
        x = F.relu(self.decoder2(x))
        return x, code

n_input=40*44
net=Autoencoder(n_input)

# Define an optimizer
learning_rate=0.01
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# create a loss function
criterion = nn.MSELoss()





