# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:09:56 2019

Normalize factors 

@author: Jincheng
"""

import copy
import numpy as np

def normalize(data, split_idx, seq_len):
    data_norm = copy.deepcopy(data)
    train = data.iloc[:split_idx, :] #where train x ends
#    tep_y = data.y
    for col in train.columns:
        train_mean = train[col].values.mean()
        train_std = train[col].values.std()
        
        tep_col = (data[col]-train_mean)/train_std
        bound = 2*train_std
        
        tep_col[tep_col > bound] = bound
        tep_col[tep_col < -bound] = -bound
        
        data_norm[col] = tep_col
  #  data_norm.y = tep_y   
    return data_norm





