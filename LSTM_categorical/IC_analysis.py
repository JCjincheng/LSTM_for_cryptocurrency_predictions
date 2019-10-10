# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:46:15 2019

Information Coefficients

@author: Jincheng Xu
"""

import pandas as pd

def combine_dataframe(data, symbol_list, split_idx, seq_len, pred_len):
    Xtrain_data = dict()
    ytrain_data = dict()
    
    first_index = split_idx-(split_idx-seq_len-1)//pred_len*pred_len
    last_index = split_idx
    for key in symbol_list:
        Xtrain_data[key] = data[key].iloc[range(first_index, last_index, pred_len), :-1]
        ytrain_data[key] = data[key].iloc[range(first_index, last_index, pred_len), -1]
    Xtrain = pd.concat(Xtrain_data)
    ytrain = pd.concat(ytrain_data)
    return Xtrain, ytrain
    
def factors_IC(Xtrain, ytrain):
    factor_keys = Xtrain.columns
    factor_ICs = dict()
    
    for key in factor_keys:
        factor_ICs[key] = Xtrain[key].corr(ytrain)   
    
    ranked_ICs = sorted(factor_ICs.items(), key=lambda d: abs(d[1]), reverse=True)
    
    return ranked_ICs