# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:54:12 2019

what factors to include, create factor object

@author: Jincheng
"""

import LSTM_categorical.factors_helper as fach
#import numpy as np

def formulate_bars(bars_df,cycle):
    bars_cycle = bars_df.resample('{0}T'.format(cycle)).agg({'open'   :'first',
                                                             'high'   :'max',
                                                             'low'    :'min',
                                                             'close'  :'last',
                                                             'volume' :'sum'
                                                            })
    return bars_cycle

#factor_name, variable analyzing, cycle
    
def add_factors(fac_list,data_df):
    for factor in fac_list:
        num = len(factor)
        if num==1:
            data_df[factor[0]] = getattr(fach, factor[0])(data_df)
        else:
            if isinstance(factor[1], int):
                data_df[str(factor[0] + str(factor[1]))] = getattr(fach, factor[0])(data_df, factor[1])
            else:
                data_df[str(factor[0] + str(factor[1]))] = getattr(fach, factor[0])(data_df, *factor[1])
    #data_df.replace([np.inf, -np.inf], np.nan, inplace = True)
    data_df.dropna(axis=0, inplace=True)

    return data_df


