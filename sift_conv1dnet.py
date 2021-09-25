# 1-D Convolutional Neural Network for SIFT inversion.
# Architecture is loosely based on the 'encoder' part of an autoencoder.
# See https://en.wikipedia.org/wiki/Autoencoder for more details
# Date: 6/1/2021
# Author: Christopher Liu

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Defining the swish activation function
# see https://en.wikipedia.org/wiki/Swish_function
def SiLU(input):
    return input * torch.sigmoid(input)

class Conv1DNN(nn.Module):
    def __init__(self, ngauges, nsources):
        super().__init__()

        self.ngauges = ngauges # Number of input channels
        self.nsources = nsources # Output dimension, number of unit sources
       
        self.conv1 = nn.Conv1d(self.ngauges, 20, 3, padding=1)  
        self.conv2 = nn.Conv1d(20, 20, 3, padding=1)
        self.conv3 = nn.Conv1d(20, 40, 3, padding=1)
        self.conv4 = nn.Conv1d(40, 40, 3, padding=1)
        self.conv5 = nn.Conv1d(40, 80, 3, padding=1)

        self.pool = nn.MaxPool1d(2, 2) # size = 2, stride = 2
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU() # ReLU added at the end to enforce positivity
        
        self.drop = nn.Dropout(p=0.3)
        
        # Input dimension needs to be consistent with output dimension
        # of the last conv layer
        self.lin = nn.Linear(80, 40, bias=True)
        self.lin2 = nn.Linear(40,nsources, bias=True)

    def forward(self, x):
        
        # Was testing swish vs leaky relu.
        if 1:
            x = self.lrelu(self.conv1(x))
            x = self.pool(x)
            x = self.lrelu(self.conv2(x))
            x = self.pool(x)
            x = self.lrelu(self.conv3(x))
            x = self.pool(x)
            x = self.lrelu(self.conv4(x))
            x = self.pool(x)
            x = self.lrelu(self.conv5(x))
            x = self.pool(x)
        
        if 0:
            x = SiLU(self.conv1(x))
            x = self.pool(x)
            x = SiLU(self.conv2(x))
            x = self.pool(x)
            x = SiLU(self.conv3(x))
            x = self.pool(x)
            x = SiLU(self.conv4(x))
            x = self.pool(x)
            x = SiLU(self.conv5(x))
            x = self.pool(x)
        
        x = torch.squeeze(x)
        
        x = self.lin(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.relu(x)
        
        return x
    
# An example for 30 minutes, change or remove as needed
class Conv1DNN_30(nn.Module):
    def __init__(self, ngauges, nsources):
        super().__init__()

        self.ngauges = ngauges # Number of input channels
        self.nsources = nsources # Output dimension, number of unit sources
        
        self.conv1 = nn.Conv1d(self.ngauges, 20, 3, padding=1)  
        self.conv2 = nn.Conv1d(20, 20, 3, padding=1)
        self.conv3 = nn.Conv1d(20, 40, 3, padding=1)
        self.conv4 = nn.Conv1d(40, 60, 3, padding=1)

        self.pool = nn.MaxPool1d(2, 2)
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU() # ReLU added at the end to enforce positivity
        
        self.drop = nn.Dropout(p=0.3)
        
        # Input dimension needs to be consistent with output dimension
        # of the last conv layer
        self.lin = nn.Linear(60, 40, bias=True)
        self.lin2 = nn.Linear(40,nsources, bias=True)

    def forward(self, x):
        
        x = self.lrelu(self.conv1(x))
        x = self.pool(x)
        x = self.lrelu(self.conv2(x))
        x = self.pool(x)
        x = self.lrelu(self.conv3(x))
        x = self.pool(x)
        x = self.lrelu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.squeeze(x)
        
        x = self.lin(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.relu(x)
        
        return x
    
class Conv1DNN_GNSS(nn.Module):
    def __init__(self, ngauges, nsources):
        super().__init__()

        self.ngauges = ngauges # Number of input channels
        self.nsources = nsources # Output dimension, number of unit sources
           
        # try first for only z component
        self.conv1 = nn.Conv1d(self.ngauges, 64, 3, padding=1)  
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(64, 96, 3, padding=1)
        self.conv5 = nn.Conv1d(96, 96, 3, padding=1)
        self.conv6 = nn.Conv1d(96, 96, 3, padding=1)
        self.conv7 = nn.Conv1d(96, 128, 3, padding=1)
        self.conv8 = nn.Conv1d(128, 128, 3, padding=1)
        self.conv9 = nn.Conv1d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool1d(2, 2) # size = 2, stride = 2
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU() # ReLU added at the end to enforce positivity
        
        self.drop = nn.Dropout(p=0.3)
        
        # Input dimension needs to be consistent with output dimension
        # of the last conv layer
        self.lin = nn.Linear(128,64, bias=True)
        self.lin2 = nn.Linear(64,nsources, bias=True)
        
    def forward(self, x):
        

        x = self.lrelu(self.conv1(x))
        x = self.pool(x)
        x = self.lrelu(self.conv2(x))
        x = self.pool(x)
        x = self.lrelu(self.conv3(x))
        x = self.pool(x)
        x = self.lrelu(self.conv4(x))
        x = self.pool(x)
        x = self.lrelu(self.conv5(x))
        x = self.pool(x)
        x = self.lrelu(self.conv6(x))
        x = self.pool(x)
        x = self.lrelu(self.conv7(x))
        x = self.pool(x)
        x = self.lrelu(self.conv8(x))
        x = self.pool(x)
        x = self.lrelu(self.conv9(x))
        x = self.pool(x)
        
        x = torch.squeeze(x)
        
        x = self.lin(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.relu(x)
        
        return x    

