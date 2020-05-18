# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:26:17 2020

@author: Luther Ollier


_________________________________Machine learning tool________________________________________________________
____________________________________part 1_____________________________________________________
"""

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import sklearn 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

print("Pytorch version:{}".format(torch.__version__))
print("Sklearn version:{}".format(sklearn.__version__))
print("Numpy version:{}".format(np.__version__))
print("Pandas version:{}".format(pd.__version__))




"""
#_________________________________Logistic Regression__________________________________#
#                                                                                      #  
#                                                                                      #
# Overview :                                                                           #
#                                                                                      #
# - Data preprocessing : generate a dataset to classify with a sklearn generator       #
#                                                                                      #
#                                                                                      #
# - Model definition and training : create a model of Logistic Regression and train    #
# it on the dataset                                                                    #
#                                                                                      #
# - Get the accuracy                                                                   #
#                                                                                      #
#                                                                                      #
#______________________________________________________________________________________#
"""
#%% Data preprocessing 
#useful links 
#https://blog.goodaudience.com/awesome-introduction-to-logistic-regression-in-pytorch-d13883ceaa90
#https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

#generate Data for classification
X_class, Y_class=make_classification(n_samples=10000, 
                                     n_classes=3, 
                                     n_features=5,
                                     n_informative=3,
                                     random_state=0)
#splitting Data
Y_class=pd.get_dummies(Y_class)
x_class_train,x_class_test,y_class_train,y_class_test=train_test_split(X_class,Y_class)
x_class_train=torch.FloatTensor(x_class_train)
x_class_test=torch.FloatTensor(x_class_test)
y_class_train=torch.tensor(y_class_train[y_class_train.columns].values)
y_class_test=torch.tensor(y_class_test[y_class_test.columns].values)

#%%model definition and training
class LogisticRegression(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
LogReg_model=LogisticRegression(5,3)

# Define a loss function and optimizer
learning_rate=0.001
optimizer = optim.SGD(LogReg_model.parameters(), lr=learning_rate)

# create a loss function
criterion = nn.L1Loss()

#training the model
n_train_samples=x_class_train.shape[0]
n_epoch =  100
batch_size = int(n_train_samples / n_epoch)  # number of examples per batch
total_loss=[]

for epoch in range(n_epoch):
    running_loss = 0
    for b in range(batch_size):
        # get the inputs
        inputs = Variable(x_class_train[b*batch_size:b*batch_size+batch_size, :], requires_grad=True)
        labels = Variable(y_class_train[b*batch_size:b*batch_size+batch_size,:])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + loss + backward + optimize
        LR_out = LogReg_model(inputs)
        loss = criterion(LR_out, labels)    
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / batch_size))
    total_loss.append(running_loss)
    running_loss = 0.0

print('Finished Training')


#%% Accuracy on train set and test set

def clf_accuracy(x,y,model):
    """
    input: the data, the associated labels, the model
    output: the accuracy
    """
    correct=0
    for i in range(x.shape[0]):
        output=model(x[i])
        _, predicted=torch.max(output.data,0)
        _, label = torch.max(y[i], 0)
        #print(predicted,label)
        correct += (predicted == label).sum()
    return 100*(correct.float()/x.shape[0])

train_accuracy=clf_accuracy(x_class_train,y_class_train,LogReg_model)
test_accuracy=clf_accuracy(x_class_test, y_class_test, LogReg_model)

print("train accuracy : {} \n test accuracy : {}".format(train_accuracy,test_accuracy))

    



