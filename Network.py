import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

class GLU(torch.nn.Module):
    def __init__(self, dim_input):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_input)
        self.fc2 = nn.Linear(dim_input, dim_input)
    
    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)
    
class GRN(torch.nn.Module):
    def __init__(self, dim_input, dim_out = None, n_hidden = 10, dropout_r = 0.1):
        super(GRN, self).__init__()
        
        
        if(dim_out != None):
            self.skip = nn.Linear(dim_input, dim_out)
        else:
            self.skip = None
            dim_out = dim_input
        
        self.fc1 = nn.Linear(dim_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, dim_out)
        self.dropout = nn.Dropout(dropout_r)
        
        self.gate = GLU(dim_out)
        
        self.norm = nn.LayerNorm(dim_out)
        
    def forward(self, x):
        a = F.elu(self.fc1(x))
        a = self.dropout(self.fc2(a))
        
        a = self.gate(a)
        
        if(self.skip != None):
            return self.norm(self.skip(x) + a)
        return self.norm(x + a)

class VSN(torch.nn.Module):
    def __init__(self, n_variables, dim_model, dropout_r = 0.1):
        super(VSN, self).__init__()
        
        #Linear transformation of inputs into dmodel vector
        self.fc = nn.Linear(1, dim_model, bias = False)
        
        self.input_grn = GRN(dim_model, dropout_r = dropout_r)
        self.vs_grn = GRN(n_variables * dim_model * 1, dim_out = n_variables, dropout_r = dropout_r)
    
    #takes input (batch_size, seq_len, n_variables, input_size)
    def forward(self, x):
        linearised = self.fc(x.unsqueeze(-1))
        
        #flatten everything except accross batch for variable selection weights
        vs_weights = self.vs_grn(linearised.flatten(start_dim = 2)) #(batch_size, seq_len, n_variables)
        vs_weights = torch.softmax(vs_weights, dim = -1).unsqueeze(-1) #(batch_size, seq_len, n_variables, 1)
        
        #input_grn applied to every input seperatly
        input_weights = self.input_grn(linearised) #(batch_size, seq_len, n_variables, dim_model)
        
        x = torch.sum((vs_weights * input_weights), dim = 2)
        return x #returns(batch_size, seq_len, dim_model)

    
class LSTMLayer(torch.nn.Module):
    def __init__(self, dim_model, n_layers = 1, dropout_r = 0.1):
        super(LSTMLayer, self).__init__()
        self.n_layers = n_layers
        self.dim_model = dim_model
        
        self.lstm = nn.LSTM(dim_model, dim_model, num_layers = n_layers, batch_first = True)
        self.hidden = None
        
        self.dropout = nn.Dropout(dropout_r)
    
    #takes input (batch_size, seq_len, dim_model)
    def forward(self, x):
        if(self.hidden == None):
            raise Exception("Call reset() to initialise LSTM Layer")
            
            
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout(x)
        
        return x, self.hidden #returns (batch_size, seq_len, dim_model), hidden
    
    def reset(self, batch_size):
        self.hidden = (torch.zeros([self.n_layers, batch_size, self.dim_model]),
                   torch.zeros([self.n_layers, batch_size, self.dim_model]))
    
