# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:19:40 2019

Genetic programming

@author: JinchengXu

"""

import pandas as pd
from gplearn import functions, genetic
import numpy as np
from LSTM_categorical.IC_analysis import combine_dataframe
from LSTM_categorical.Normalization import normalize

def GP(data, split_idx, symbol_list, seq_len, pred_len):
        
    Xtrain, ytrain = combine_dataframe(data, symbol_list, split_idx, seq_len, pred_len)
        
    def rank(X1):
        return np.argsort(X1)
    
    rank = functions.make_function(function=rank,
                        name='rank',
                        arity=1)

    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min', rank]
   
    #self_defined_metric = fitness.make_fitness(ic, greater_is_better=True, verbose=1)

    gp = genetic.SymbolicTransformer(generations=1, population_size=1000,
                         function_set=function_set,
                         max_samples=0.9,
                         metric = 'spearman',
                         stopping_criteria = 1,
                         verbose=1,
                         random_state=0, n_jobs=3)
    gp.fit(Xtrain, ytrain)
    
    new_feature_names = []
    for i in gp:
        new_feature_names.append(str(i))
            
    new_data = dict()
    for key in symbol_list:
        gp_features = gp.transform(data[key].iloc[:,:-1])
        gp_df = normalize(pd.DataFrame(gp_features), split_idx, pred_len)
        gp_df.index = data[key].index
        gp_df.columns = new_feature_names
        new_data[key] = pd.concat([data[key].iloc[:,:-1], gp_df, data[key].iloc[:,-1]], axis = 1)
        
    return new_data
    
    
    
    
    
    
    
    
    
    