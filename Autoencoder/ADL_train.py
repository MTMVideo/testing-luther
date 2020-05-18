# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:44:36 2020

@author: Luther Ollier

_________________________________Auto-encoder on audio files_________________________________________________________
______________________________________Auto-encoder _________________________________________________

"""
import torch
from torch.utils.data import DataLoader

import numpy as np

import librosa
import librosa.display

import tqdm

from ADL_Autoencoder import audio_dataset, optimizer, net, criterion


for i,j in zip([torch,np,librosa],["torch","numpy","librosa"]):
    print("{} version :".format(j) + i.__version__)
    
#%%
"""
#_________________________________Autoencoder on AudioFile_____________________________#
#                                                                                      #  
#                                                                                      #
#                                                                                      #
# - Training :  train and save the model                                               #
#                                                                                      #
#                                                                                      #
#______________________________________________________________________________________#
"""
#%% Training

## Train the network
n=audio_dataset.__len__()
n_epoch = 100
batch_size = int(n / n_epoch)  # number of examples per batch
total_loss=[]
audio_loader = DataLoader(audio_dataset,batch_size=batch_size, shuffle=True)

for epoch in range(n_epoch):
    running_loss = 0
    for i,(inputs,label) in tqdm.tqdm(enumerate(audio_loader)):
        # get the inputs
        inputs = inputs.view(-1,44*40)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + loss + backward + optimize
        net_out,weight = net(inputs)
        loss = criterion(net_out, inputs)
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / batch_size))
    running_loss = 0.0

print('Finished Training')

## Save the weights
torch.save(net.state_dict(), 'weights_autoencoder_run2.pt')