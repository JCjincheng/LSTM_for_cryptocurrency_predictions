# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:15:43 2019

create different datasets, the interval between each time interval is always 30 days

@author: Jincheng Xu
"""

from LSTM_categorical.LSTM import data_analysis
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
from LSTM_categorical.clean_data import read_in_file, subdata_by_datetime

def strategy_return(stg_df, TxnFees):
    #return
    stg_df['position'] = stg_df['pred'].apply(lambda x: 1 if x>0.5 else -1 )
    returnSeries = stg_df['real']*stg_df['position']

    #add txnfees
    stg_df['diff'] = abs(stg_df['position'].diff())
    stg_df.loc[stg_df.index[0], 'diff'] = abs(stg_df.loc[stg_df.index[0], 'position']) #初始开仓手续费
    tradecount_series = stg_df['position']*stg_df['diff']
    tradecount_series[tradecount_series != 0] = 1
    tradecount = tradecount_series.sum()    
    stg_df.loc[stg_df.index[-1], 'diff'] = stg_df.loc[stg_df.index[-1], 'diff'] + abs(stg_df.loc[stg_df.index[-1], 'position']) #最终平仓手续费
    stg_df['txnfees'] = stg_df['diff']*TxnFees
    returnSeries = returnSeries-stg_df['txnfees']
    return returnSeries, tradecount
    
def portfolio_return(stg_dict, TxnFees ):
    return_dict = dict()
    tradecounts_dict = dict()
    for symbol, stg_df in stg_dict.items():
        return_dict[symbol], tradecounts_dict[symbol] = strategy_return(stg_df, TxnFees)            
    return_df = pd.DataFrame(return_dict)
    return_final = return_df.mean(axis = 1)
    return return_final, tradecounts_dict
    
def for_diff_cycle(cycle, seq_len):    
    
    total_data = dict()
    
    symbol_list = ['btc.usdt.csv', 'eth.usdt.csv', 'eos.usdt.csv', 'xrp.usdt.csv', 'ltc.usdt.csv']
    
    #the path oFf file you are reading in
    path = 'E:\\jccassie\data_for_lstm'
    
    #read in file
    for key in symbol_list:
        total_data[key] = read_in_file(key, path)
        
    #begin and end datetime
    #subset by begin and end date
    #date format: '2018-01-01 00:00:00'
    begin_time = '2018-01-02 00:00:00'
    end_time = '2019-01-01 23:59:00'
           
    #model_performance
    train_test_interval, loss, acc, stg_dict = data_analysis(total_data, begin_time, end_time, symbol_list, cycle, seq_len)
    
    sec_begin_time = str(datetime.strptime(begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    sec_end_time = str(datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    
    sec_train_test_interval, sec_loss, sec_acc, sec_stg_dict = data_analysis(total_data, sec_begin_time, sec_end_time, symbol_list, cycle, seq_len)
     
    trd_begin_time = str(datetime.strptime(sec_begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    trd_end_time = str(datetime.strptime(sec_end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    
    trd_train_test_interval, trd_loss, trd_acc, trd_stg_dict = data_analysis(total_data, trd_begin_time, trd_end_time, 
                                                                    symbol_list, cycle, seq_len)

    forth_begin_time = str(datetime.strptime(trd_begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    forth_end_time = str(datetime.strptime(trd_end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    
    forth_train_test_interval, forth_loss, forth_acc, forth_stg_dict = data_analysis(total_data, forth_begin_time, 
                                                                        forth_end_time, symbol_list
                                                                        , cycle, seq_len)

   
    fifth_begin_time = str(datetime.strptime(forth_begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    fifth_end_time = str(datetime.strptime(forth_end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
     
    fifth_train_test_interval, fifth_loss, fifth_acc, fifth_stg_dict = data_analysis(total_data, fifth_begin_time, 
                                                                        fifth_end_time, 
                                                                        symbol_list, cycle, seq_len)
    
    six_begin_time = str(datetime.strptime(fifth_begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    six_end_time = str(datetime.strptime(fifth_end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
     
    six_train_test_interval, six_loss, six_acc, six_stg_dict = data_analysis(total_data, six_begin_time, 
                                                                        six_end_time, 
                                                                        symbol_list, cycle, seq_len)

    seventh_begin_time = str(datetime.strptime(six_begin_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
    seventh_end_time = str(datetime.strptime(six_end_time, '%Y-%m-%d %H:%M:%S') + train_test_interval)
     
    seventh_train_test_interval, seventh_loss, seventh_acc, seventh_stg_dict = data_analysis(total_data, 
                                                                        seventh_begin_time, 
                                                                        seventh_end_time, 
                                                                        symbol_list, cycle, seq_len)
    #caculate returns
    return_final, tradecounts = portfolio_return(stg_dict, 0)
    sec_return_final, sec_tradecounts = portfolio_return(sec_stg_dict, 0)
    trd_return_final, trd_tradecounts = portfolio_return(trd_stg_dict, 0)
    forth_return_final, forth_tradecounts = portfolio_return(forth_stg_dict, 0)
    fifth_return_final, fifth_tradecounts = portfolio_return(fifth_stg_dict, 0)
    sixth_return_final, six_tradecounts = portfolio_return(six_stg_dict, 0)
    seventh_return_final, seventh_tradecounts = portfolio_return(seventh_stg_dict, 0)

    #join all dataframes
    all_return_df = pd.concat([return_final, sec_return_final, trd_return_final, 
                               forth_return_final, fifth_return_final, sixth_return_final,
                               seventh_return_final
    ])
    
    all_tradecounts_df = pd.DataFrame([tradecounts, sec_tradecounts, trd_tradecounts,
                                       forth_tradecounts, fifth_tradecounts,
                                       six_tradecounts, seventh_tradecounts]).T
    
    
    TxnFees = 0.0005
    return_final_txn, tradecounts_txn = portfolio_return(stg_dict, TxnFees)
    sec_return_final_txn, sec_tradecounts_txn = portfolio_return(sec_stg_dict, TxnFees)
    trd_return_final_txn, trd_tradecounts_txn = portfolio_return(trd_stg_dict, TxnFees)
    forth_return_final_txn, forth_tradecounts_txn = portfolio_return(forth_stg_dict, TxnFees)
    fifth_return_final_txn, fifth_tradecounts_txn = portfolio_return(fifth_stg_dict, TxnFees)
    six_return_final_txn, six_tradecounts_txn = portfolio_return(six_stg_dict, TxnFees)
    seventh_return_final_txn, seventh_tradecounts_txn = portfolio_return(seventh_stg_dict, TxnFees)
    
    all_return_df_txn = pd.concat([return_final_txn, sec_return_final_txn, trd_return_final_txn, 
                               forth_return_final_txn, fifth_return_final_txn, 
                               six_return_final_txn, seventh_return_final_txn
    ])
    
    acc_df = pd.DataFrame([acc, sec_acc, trd_acc, forth_acc, fifth_acc, six_acc, seventh_acc])
    return all_return_df.cumsum(), all_return_df_txn.cumsum(), acc_df, all_tradecounts_df

if __name__ == '__main__':
    
    seed(1)
    set_random_seed(2)
    
    # cycle = i, seq_len = j
    for i in [75]:
        for j in [20]:
            cum_return_75, cum_return_txn_75, acc_75, tradecounts = for_diff_cycle(i, j)
            cum_return_75.to_csv('E:/jccassie/LSTM_categorical/'+str(i)+'_cycle'+str(j)+'_seq'+'.csv')
            cum_return_txn_75.to_csv('E:/jccassie/LSTM_categorical/'+str(i)+'txn_cycle'+str(j)+'_seq'+'.csv')
            acc_75.to_csv('E:/jccassie/LSTM_categorical/'+str(i)+'acc_cycle120'+str(j)+'_seq' +'.csv')
            tradecounts.to_csv('E:/jccassie/LSTM_categorical/'+str(i)+'tradecounts_'+str(i)+'_cycle'+str(j)+'seq' +'.csv')
# =============================================================================
#     
#     path = 'E:\\jccassie\\LSTM\\'
#     
# # =============================================================================
# #     tep=[]
# #     for i in [15]:
# #         tep.append(pd.read_csv(path + 'accuracy_idx_' + str(i) + '.csv').iloc[:,1])
# #     
# #     acc_df = pd.concat(tep, axis=1)
# #     acc_df.columns = [60,75]
# #     acc_df = acc_df.append(acc_df.mean(axis=0), ignore_index=True)
# #     acc_df['mean_for_diff_months'] = acc_df.mean(axis=1)
# #     
# #     acc_df.index = ['Jan', 'Feb', 'Mar',
# #                     'Apr', 'May', 'mean_for_diff_cycle']
# #  
# #     acc_df
# # =============================================================================
#      
#     for i in [5,10]:
#         pd.read_csv(path + 'cycle_15seq_' + str(i) + '.csv').plot()
#     
#     for i in [5,10]:
#         pd.read_csv(path + 'txn_cycle_15seq_' + str(i) + '.csv').plot()
#      
#  
# 
# =============================================================================
