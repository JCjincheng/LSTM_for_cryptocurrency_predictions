# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:36:15 2019

prepare train and test datasets for model use

@author: Jincheng
"""

import math
import numpy as np

def split_time(datdict, split_perc, pred_len):
    tep_data = next(iter(datdict.values()))
    split_idx = len(tep_data)-math.ceil(len(tep_data)*(1-split_perc)/pred_len)*pred_len-1
    #split_idx = math.floor(len(tep_data)*split_perc) #where first test x starts
    return split_idx, tep_data.index[split_idx-pred_len]

def load_data(train_data, first_idx, last_idx, seq_len, pred_len):
    data_array=train_data.values
    res = []
    for i in range(first_idx, last_idx+pred_len, pred_len):
        res.append(data_array[(i-seq_len+1):(i+1)]) #last 31504
    return res

#combine 6 different cryptos
def train_test(datdict, split_idx, seq_len, pred_len):
    combined_data_train = []
    combined_data_test = []
    first_index = split_idx-(split_idx-seq_len-1)//pred_len*pred_len #22
    last_index = split_idx
    for i in datdict.values():
        combined_data_train = combined_data_train + load_data(i, first_index, last_index-pred_len, seq_len, pred_len)
        combined_data_test = combined_data_test + load_data(i, last_index, len(i)-1, seq_len, pred_len)
    
    train = np.array(combined_data_train)
    test = np.array(combined_data_test)
    
    X_train = train[:,:,:-1]
    y_train = train[:,-1,-1]
    
    X_test = test[:,:,:-1]
    y_test = test[:,-1,-1]
    
    return X_train, y_train, X_test, y_test




