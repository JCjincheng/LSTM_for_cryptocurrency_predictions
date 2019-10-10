# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:05 2019

customize y

@author: Jincheng Xu
"""

def act_return(a):
    return (a[-1]/a[0] - 1)

def up_and_down(a):
    return (1 if a[-1]>=a[0] else 0)

#add y_factor
def add_y(data, pred_len, name):
    data['returns'] = data['close'].rolling(pred_len+1).apply(act_return).shift(-pred_len)
    data[name] = data['close'].rolling(pred_len+1).apply(up_and_down).shift(-pred_len)
    #data['y_cat'] = data['close'].rolling(pred_len+1).apply(up_and_down).shift(-pred_len)
    last_col = data.pop(name)
    #sec_last_col = data.pop('y_cat')
    #data['y_cat'] = sec_last_col
    data[name] = last_col
    return data.dropna()
    
