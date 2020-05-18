# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:04:29 2020

@author: Luther Ollier


_________________________________Machine learning tool________________________________________________________
____________________________________part 2_____________________________________________________
"""

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



print("Pytorch version:{}".format(torch.__version__))

"""
#_________________________________MLP for digits classification________________________#
#                                                                                      #  
#                                                                                      #
# Overview :                                                                           #              
#                                                                                      #                                                            #
#                                                                                      #
# - Data preprocessing : loading the test and train dataset                            #
#                                                                                      #
#                                                                                      #
# - Model definition : create a 3 layer network with 250 neurones for each hidden layer#
#                                                                                      #
#                                                                                      #
# - Training :  train and save the model                                               #
#                                                                                      #
#                                                                                      #
# - Accuracy                                                                           #
#                                                                                      #
#______________________________________________________________________________________#
"""
#%%Data preprocessing

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

n_train_samples=len(train_dataset)
n_test_samples=len(test_dataset)
n_epoch =  1000
batch_size = int(n_train_samples / n_epoch)  # number of examples per batch

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)


    
#%%Model creation

n_hidden=250

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.fc1=nn.Linear(in_features,n_hidden)
        self.fc2=nn.Linear(n_hidden,n_hidden)
        self.fc3=nn.Linear(n_hidden,out_features)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

digit_clf=MLP(784,10)

# Define a loss function and optimizer
learning_rate=0.001
optimizer = optim.Adam(digit_clf.parameters(), lr=learning_rate)

# create a loss function
criterion = nn.CrossEntropyLoss()

#%% Training

total_loss=[]
for epoch in range(n_epoch):
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # get the inputs
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + loss + backward + optimize
        clf_out = digit_clf(images)
        loss = criterion(clf_out, labels)    
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / batch_size))
    total_loss.append(running_loss)
    running_loss = 0.0

print('Finished Training')

## Save the weights
torch.save(digit_clf.state_dict(), 'weights.pt')

#%%Accuracy 

model = MLP(784,10)
model.load_state_dict(torch.load('weights.pt'))
model.eval()

## Training accuracy

def clf_accuracy(dataset,model):
    """
    input: the pytorch dataset , the model
    output: the accuracy
    """
    dataloader=DataLoader(dataset=dataset,
                          batch_size=1)
    correct=0
    for i, (image,label) in enumerate(dataloader):
        output=model(image.view(-1,784))
        _, predicted=torch.max(output.data,1)
        correct += (predicted == label).sum()
    return 100*(correct.float()/len(dataset))

train_accuracy=clf_accuracy(train_dataset,model)
test_accuracy=clf_accuracy(test_dataset, model)

print("train accuracy : {} \ntest accuracy : {}".format(train_accuracy,test_accuracy))












