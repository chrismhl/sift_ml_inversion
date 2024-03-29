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
    def __init__(self, ngauges, nsources, eta_us):
        super().__init__()

        self.ngauges = ngauges # Number of input channels
        self.nsources = nsources # Output dimension, number of unit sources
        self.usts = eta_us.requires_grad_(False) # unit source response
        if 0:
            # try first for only z component
            self.conv1 = nn.Conv1d(self.ngauges, 96, kernel_size=3, padding=1, bias = False)  
            self.conv2 = nn.Conv1d(96, 96, kernel_size=3, padding=1, bias = False)
            self.conv3 = nn.Conv1d(96, 96, kernel_size=3, padding=1, bias = False)
            self.conv4 = nn.Conv1d(96, 96, kernel_size=3, padding=1, bias = False)
            self.conv5 = nn.Conv1d(96, 128, kernel_size=3, padding=1, bias = False)
            self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias = False)
            self.conv7 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias = False)
            self.conv8 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias = False)
            self.conv9 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias = False)

            self.pconv1 = nn.Conv1d(128,64, kernel_size=1, padding=0, stride =1)
            self.pconv2 = nn.Conv1d(64,nsources, kernel_size=1, padding=0, stride =1)

            self.batch64_1 = nn.BatchNorm1d(96)
            self.batch64_2 = nn.BatchNorm1d(96)
            self.batch64_3 = nn.BatchNorm1d(96)
            self.batch96_1 = nn.BatchNorm1d(96)
            self.batch96_2 = nn.BatchNorm1d(128)
            self.batch96_3 = nn.BatchNorm1d(128)
            self.batch128_1 = nn.BatchNorm1d(128)
            self.batch128_2 = nn.BatchNorm1d(128)
            self.batch128_3 = nn.BatchNorm1d(128)
        
        if 0:
            self.conv1 = nn.Conv1d(self.ngauges, 96, kernel_size=3, padding=1) 
            self.sconv1 = nn.Conv1d(96, 96, kernel_size=4, stride = 4, padding=0)
            self.conv2 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.sconv2 = nn.Conv1d(96, 96, kernel_size=4, stride = 4, padding=0)
            self.conv3 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.sconv3 = nn.Conv1d(96, 96, kernel_size=2, stride = 2, padding=0)
            self.conv4 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.sconv4 = nn.Conv1d(96, 96, kernel_size=2, stride = 2, padding=0)
            self.conv5 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
            self.sconv5 = nn.Conv1d(128, 128, kernel_size=2, stride = 2, padding=0)
            self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.sconv6 = nn.Conv1d(128, 128, kernel_size=2, stride = 2, padding=0)
            self.conv7 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.sconv7 = nn.Conv1d(128, 128, kernel_size=2, stride = 2, padding=0)

            self.batch1 = nn.BatchNorm1d(96)
            self.batch2 = nn.BatchNorm1d(96)
            self.batch3 = nn.BatchNorm1d(96)
            self.batch4 = nn.BatchNorm1d(96)
            self.batch5 = nn.BatchNorm1d(128)
            self.batch6 = nn.BatchNorm1d(128)
            self.batch7 = nn.BatchNorm1d(128)
            
            self.pconv1 = nn.Conv1d(128,64, kernel_size=1, padding=0, stride =1)
            self.pconv2 = nn.Conv1d(64,nsources, kernel_size=1, padding=0, stride =1)
            
        if 1:
            self.conv1 = nn.Conv1d(self.ngauges, 96, kernel_size=3, padding=1)  
            self.conv2 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(96, 96, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
            self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.conv7 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.conv8 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.conv9 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

            self.pconv1 = nn.Conv1d(128,64, kernel_size=1, padding=0, stride =1)
            self.pconv2 = nn.Conv1d(64,nsources, kernel_size=1, padding=0, stride =1)
        
        self.pool = nn.MaxPool1d(2, 2) # size = 2, stride = 2
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU() # ReLU added at the end to enforce positivity
        
        self.drop = nn.Dropout(p=0.5)
        
        self.lin = nn.Linear(128,64, bias=True)
        self.lin2 = nn.Linear(64,nsources, bias=True)
        
    def forward(self, x):
       
        if 0: # this sucks
            x = self.lrelu(self.batch64_1(self.conv1(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch64_2(self.conv2(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch64_3(self.conv3(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch96_1(self.conv4(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch96_2(self.conv5(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch96_3(self.conv6(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch128_1(self.conv7(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch128_2(self.conv8(x)))
            x = self.pool(x)
            x = self.lrelu(self.batch128_3(self.conv9(x)))
            x = self.pool(x)
        
        # Wacky one
        if 0:
            x = self.lrelu(self.batch1(self.conv1(x)))
            x = self.sconv1(x)
            x = self.lrelu(self.batch2(self.conv2(x)))
            x = self.sconv2(x)
            x = self.lrelu(self.batch3(self.conv3(x)))
            x = self.sconv3(x)
            x = self.lrelu(self.batch4(self.conv4(x)))
            x = self.sconv4(x)
            x = self.lrelu(self.batch5(self.conv5(x)))
            x = self.sconv5(x)
            x = self.lrelu(self.batch6(self.conv6(x)))
            x = self.sconv6(x)
            x = self.lrelu(self.batch7(self.conv7(x)))
            x = self.sconv7(x)
            
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
            x = self.lrelu(self.conv6(x))
            x = self.pool(x)
            x = self.lrelu(self.conv7(x))
            x = self.pool(x)
            x = self.lrelu(self.conv8(x))
            x = self.pool(x)
            x = self.lrelu(self.conv9(x))
            x = self.pool(x)
        
        x = self.pconv1(x)
        x = self.pconv2(x)
        x = torch.squeeze(x)
        x = torch.transpose(self.relu(x),0,1)
        
        x = torch.stack([torch.transpose(torch.matmul(self.usts[:,0,:], x),0,1),\
                        torch.transpose(torch.matmul(self.usts[:,1,:], x),0,1),\
                        torch.transpose(torch.matmul(self.usts[:,2,:], x),0,1)],\
                       dim = 1)  
        return x    

class Conv1DNN_GNSS_RNN(nn.Module):
    def __init__(self, ninput, noutput, layer_dim, hidden_dim, eta_us):
        
        super().__init__()
        self.ninput = ninput # number of input sensors
        self.noutput = noutput # dim output weights
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.usts = eta_us.requires_grad_(False) # unit source response
        
        # LSTM/RNN
        self.rnn = nn.LSTM(input_size = self.ninput, hidden_size = self.hidden_dim,\
                           num_layers = self.layer_dim, batch_first=True, dropout = 0.3)
        
        # output block
        self.lin = nn.Linear(self.hidden_dim, int(np.round(self.hidden_dim/2)), bias=True)
        self.lin2 = nn.Linear(int(np.round(self.hidden_dim/2)), self.noutput, bias = True)
        self.relu = nn.ReLU() 

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()
        
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.lin(out[:, -1, :])
        out = self.lin2(out)
        out = self.relu(out)
        
        if 0:
            out = torch.transpose(self.relu(out),0,1)
        
            out = torch.stack([torch.transpose(torch.matmul(self.usts[:,0,:], out),0,1),\
                        torch.transpose(torch.matmul(self.usts[:,1,:], out),0,1),\
                        torch.transpose(torch.matmul(self.usts[:,2,:], out),0,1)],\
                       dim = 1)    
        
        return out