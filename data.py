import pandas as pd
import datetime
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
import numpy as np

import random

import plotly.graph_objects as go

header = {'open_time' : 1, 'open' : 2, 'high' : 3, 'low' : 4, 'close' : 5}#, 'volume' : 6}

#load_data from file
def load_data(year, symbol, interval = "1m"):
    if(type(year) == type([])):
        frames = []
        for y in year:
            frames.append(pd.read_csv("data_{}/{}_{}.csv".format(interval, y, symbol), header = None)[list(header.values())])
        data_ = pd.concat(frames)
        #del frames                  
    else:
        print("sdf")
        data_ = pd.read_csv("data/{}_{}.csv".format(year, symbol), header = None)[list(header.values())]
    
    #convert timestamp into month and day numbers
    data_['hour'] = pd.to_datetime(data_[1], unit='ms').apply(lambda x: x.hour)    
    data_['day'] = pd.to_datetime(data_[1], unit='ms').apply(lambda x: x.day - 1)
    data_['month'] = pd.to_datetime(data_[1], unit='ms').apply(lambda x: x.month - 1)
    print("done")
    return (data_, symbol, year)

def get_batches(data_, in_seq_len, out_seq_len, batch_size, epochs = 1, random_index = True, gpu = True, normalise = True, increment = 1):
    data = data_[0].copy()
    symbol = data_[1]
    
    if normalise:
        data[[2,3,4,5]] = (data[[2,3,4,5]] - data[[2,3,4,5]].stack().mean()) / data[[2,3,4,5]].stack().std()
    #data = data.drop(data.index[0])    
    
    in_seq_continuous = []
    in_seq_discrete = []
    out_seq = []
    target_seq = []
    
    ba = 0
    idx = 0
    
    while True:
        if(random_index):
            i = random.randrange(1, data.shape[0] - (in_seq_len + out_seq_len + 1))
        else:
            i = idx
            idx += increment
        
        batch_data = data.iloc[i:i + in_seq_len + out_seq_len].copy()
        
        #normalise batches
        if normalise:
            a = batch_data.iloc[0:in_seq_len, [1 ,2, 3, 4]]
            #a = (a - a.iloc[-1]) / a.stack().std()
            std = a.stack().std()
            a = (a - a.iloc[-1]) / std
            batch_data.iloc[0:in_seq_len, [1 ,2, 3, 4]] = a
            
            b = batch_data.iloc[in_seq_len:in_seq_len + out_seq_len,[header['close']-1]]
            b = (b - b.iloc[0]) / std #a.iloc([-out_seq_len:]).std()
            batch_data.iloc[in_seq_len:in_seq_len + out_seq_len,[header['close']-1]] = b
        

            
            
        in_seq_continuous.append(batch_data.iloc[0:in_seq_len, [1 ,2, 3, 4]].values)
        in_seq_discrete.append(batch_data.iloc[0:in_seq_len, [5, 6, 7]].values)
        out_seq.append(batch_data.iloc[in_seq_len:in_seq_len + out_seq_len, [5, 6, 7]].values)
        target_seq.append(batch_data.iloc[in_seq_len:in_seq_len + out_seq_len,[header['close']-1]].values)
        
        ba += 1
        if(ba >= batch_size):
            #[(batch_size, in_seq_len, 4), (batch_size, in_seq_len, 3), (batch_size, out_seq_len, 3), (batch_size, out_seq_len, 1)]
            if(not gpu):
                dtype = torch.FloatTensor
            else:
                dtype = torch.cuda.FloatTensor
                
            yield (torch.tensor(in_seq_continuous).type(dtype).unsqueeze(-1),
                   torch.tensor(in_seq_discrete).type(dtype),
                   torch.tensor(out_seq).type(dtype),
                   torch.tensor(target_seq).type(dtype))
            
            in_seq_continuous = []
            in_seq_discrete = []
            out_seq = []
            target_seq = []
            ba = 0
            
def one_hot(x, dims, gpu = True):
    out = []
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    
    if(not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor
        
    for i in range(0, x.shape[-1]):
        x_ = x[:,:,i].byte().cpu().long().unsqueeze(-1)
        o = torch.zeros([batch_size, seq_len, dims[i]]).long()

        o.scatter_(-1, x_,  1)
        out.append(o.type(dtype))
    return out