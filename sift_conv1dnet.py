# 1-D Convolutional Neural Network for SIFT inversion.
# Architecture is loosely based on the 'encoder' part of an autoencoder.
# See https://en.wikipedia.org/wiki/Autoencoder for more details
# Date: 5/28/2021
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

        self.pool = nn.MaxPool1d(2, 2)
        
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
    

