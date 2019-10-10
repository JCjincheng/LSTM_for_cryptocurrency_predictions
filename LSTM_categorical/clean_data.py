# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:54:12 2019

Read in data, and clean it

@author: Jincheng
"""

import numpy as np
import pandas as pd

# this function returns the column index of nan values and inf values
def read_in_file(file, path):
    data = pd.read_csv(path + '\\' + file, index_col=0, parse_dates = True)
    return data
    
def subdata_by_datetime(data, begin, end):
    tep = data.loc[begin:end]
    return tep
   
def cleaned_dataset(data):
    data.replace([np.inf, -np.inf], np.nan)
    
    idx = np.where(data.isnull().any())
    print (idx)
    
    #code to deal with nan/inf value goes here
    
    #如果volume是0， 改为上一个volume 的值
    #data['volume'] = data['volume'].replace(to_replace = 0, method='ffill').values
    
    return (data)

# how to call example
# cleaned_dataset('btc.usdt/okex',datetime(2018,1,1),datetime(2018,1,25))
    

    