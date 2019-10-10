# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:19:46 2019

deep learning model, output the predicted value

@author: Jincheng
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import optimizers
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping

from LSTM_categorical.customize_y import add_y
from LSTM_categorical.data_for_model import split_time, train_test
from LSTM_categorical.clean_data import cleaned_dataset,read_in_file, subdata_by_datetime
from LSTM_categorical.factors_generator import formulate_bars, add_factors
from LSTM_categorical.Normalization import normalize
from LSTM_categorical.genetic_programming import GP
from LSTM_categorical.IC_analysis import combine_dataframe, factors_IC
import keras as K
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

#from sklearn.decomposition import PCA

def build_learning_model(layers,neurons,d):
    my_init = K.initializers.glorot_uniform(seed=1)
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), 
                   activation='relu', return_sequences=True, kernel_initializer=my_init))
# =============================================================================
#     model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), 
#                    activation='relu', return_sequences=True))
# =============================================================================
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], return_sequences=True, kernel_initializer=my_init))
    #model.add(LSTM(neurons[1], return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[2], return_sequences=False, kernel_initializer=my_init))
    #model.add(LSTM(neurons[2], return_sequences=False))
    model.add(Dropout(d))

    #model.add(Dense(neurons[1], activation='tanh'))
    model.add(Dense(neurons[3],  activation='softmax', kernel_initializer=my_init))
    #model.add(Dense(neurons[3],  activation='sigmoid')) 
    #softmax for multiclass, sigmoid for binary
    # model = load_model('my_LSTM_stock_model1000.h5')
    #adam = keras.optimizers.Adam(decay=0.2, lr=0.01)
    opt = optimizers.RMSprop(lr=0.001)
    #, metrics=['accuracy']

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.summary()
    return model

# =============================================================================
# def pred_return(y_test, y_pred):
#     y_pred = y_pred.reshape(-1,)
#     pred_return = []
#     for i in range(len(y_test)):
#         if y_pred[i]>=0.0003:
#             pred_return.append(y_test[i])
#         elif y_pred[i]<=-0.0003:
#             pred_return.append(-y_test[i])
#         else:
#             pred_return.append(0)
#     return (pred_return)
# =============================================================================
    
def valid_acc(y_test, y_pred):
    if len(y_test)!=len(y_pred): return "something wrong"
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    label,counts = np.unique(y_pred==y_test, return_counts=True)
    return counts[1]/(counts[0]+counts[1])

def data_analysis(total_data, begin_time, end_time, symbol_list, cycle, seq_len):   
#if __name__ == '__main__':
# =============================================================================
#     
#     begin_time = '2018-02-02 00:00:00'
#     end_time = '2019-02-01 23:59:00'
#     
# =============================================================================
    #可改参数
    #cycle = 75                           #read in interval, ex: read in data in 15 mins each
    #seq_len    = 5                       #X length
    pred_len   = 1                       #prediction length, y = 8th/1st - 1
    win_len    = seq_len + pred_len      #length of each window
    neurons    = [32,32,16,2]
    
    split_perc = 0.9                     #train and test split at what percentage
    d          = 0.2                     #dropout pencentage in model, to reduce overfitting 
    epochs     = 30                      #how many runs we want, depends on when does it converges
    batch_size = 1500                    #sample size for each learning 
    
    
    #Factors we mean to add to the dataset physically
    self_defined_factors = [
        ['norm_ma', 10],
        ['norm_mv', 15],
        ['bar_ratio'], ['bias',(10,20)], ['momentum', 20], 
        ['AD'], ['aroon_argmax', 20], ['aroon_argmin', 20],
        ['alpha009', 20], ['rsi', 20], ['rc', 20],
        ['vr',20], ['pvt',20], ['cr',20], ['wvad',20],
        ['adtm', 20]
    ]

    data_factors_including = ['close', 'volume']
    
   # feature_num = len(FACTORS)
    
    #enter the csv file name 
    #symbol_list = ['btc.csv', 'eth.csv', 'eos.csv', 'xrp.csv', 'ltc.csv']
    #symbol_list = ['btc.csv', 'eth.csv']
  
    data = dict()
    for key in symbol_list:
        data[key] = subdata_by_datetime(total_data[key], begin_time, end_time)
  
    #clean data
    for key in symbol_list:
        data[key] = cleaned_dataset(data[key])
    
    #change to 15mins data
    for key in symbol_list:
        data[key] = formulate_bars(data[key], cycle)
        
    colnames = next(iter(data.values())).columns
    del_colnames = np.setdiff1d(colnames, data_factors_including).tolist()
    
    #add factors
    for key in symbol_list:    
        data[key] = add_factors(self_defined_factors, data[key])
        
    #add return 
    returns = dict()
    for key in symbol_list:
        #data[key] = add_y(data[key], pred_len, 'returns')
        data[key] = add_y(data[key], pred_len, 'y')
        returns[key] = data[key].returns
        del (data[key])['returns']
        
    end_idx = next(iter(data.values())).index[-1]
    plot_time_series_idx = next(iter(data.values())).index #indices of entire dataset
    
    #subtract average return
# =============================================================================
#     all_returns = []
#     for key in symbol_list:
#         all_returns.append(data[key].y)
#     all_returns = pd.concat(all_returns, axis=1)
#     mean_returns = all_returns.mean(axis=1)
#     
#     for key in symbol_list:
#         data[key]['y'] = data[key].y - mean_returns
# 
# =============================================================================
    #actual train and test split point
    split_idx, start_test_time = split_time(data, split_perc, pred_len) #split_idx is the length of train, train'y can not use the data of test
    train_test_interval = end_idx - start_test_time

    #Normalize: change to standard normal distribution
    for key in symbol_list: #0~31505
        data[key].iloc[:,:-1] = normalize(data[key].iloc[:,:-1], split_idx, pred_len)
        data[key] = data[key].drop(del_colnames,axis=1)
    
    #check factors ICs
# =============================================================================
#     Xtrain, ytrain = combine_dataframe(data, symbol_list, split_idx, seq_len, pred_len) #21~31503
#     rank = factors_IC(Xtrain, ytrain) 
#     
# =============================================================================
    #each_data_length = len(next(iter(data.values())))
    
    #genetic programming
    each_data_length = len(next(iter(data.values())))  
    
    #data = GP(data, split_idx, symbol_list, seq_len, pred_len)
    #check IC again 
    Xtrain_after_GP, ytrain_after_GP = combine_dataframe(data, symbol_list, split_idx, seq_len, pred_len)
    rank_GP = factors_IC(Xtrain_after_GP, ytrain_after_GP)
    #print(next(iter(data.values())).corr())
    print (rank_GP)
# =============================================================================
#     
#     pca = PCA(n_components=5)
#     pca.fit(Xtrain_after_GP)
#     
#  #   explained_variance = pca.explained_variance_ratio_
#     
#     pca_x = pd.DataFrame(pca.transform(pd.concat(data.values()).iloc[:, :-1]))
#     #print(pca_x.shape)
#     scaler_pca = normalize(pca_x, split_idx, seq_len)
#     scaler_pca['y'] = pd.concat(data.values()).y.values  
#     
#     for i in range(len(symbol_list)):
#         data[symbol_list[i]] = scaler_pca[each_data_length*i:each_data_length*(i+1)]
# =============================================================================
    
    #drop factors
    factors_to_keep = []
    for i in rank_GP:
        if abs(i[1])>=0.02:
            factors_to_keep.append(i[0])
    print (factors_to_keep)
    for key in symbol_list:
        data[key] = data[key].loc[:, factors_to_keep+['y']]
    
    #六币合一, 划分train， test
    X_train, y_train, X_test, y_test = train_test(data, split_idx, seq_len, pred_len)
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    encoder_test = LabelEncoder()
    encoder_test.fit(y_test)
    encoded_Y_test = encoder_test.transform(y_test)
    dummy_y_test = np_utils.to_categorical(encoded_Y_test)
    #模型e33333333333
    shape        = [X_train.shape[2], seq_len, 2]  # feature, seq_len, output_dim
    ml_model  = build_learning_model(shape,neurons,d)
    callback = EarlyStopping(monitor="loss", patience=5, verbose=1, mode="min")

    ml_model.fit(
        X_train,
        dummy_y,
        batch_size=batch_size,
        epochs=epochs,    
        verbose=1,
        validation_split=0.1,
        callbacks=[callback])    

    loss, acc = ml_model.evaluate(X_test, dummy_y_test, batch_size=batch_size)
    y_pred = ml_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    #dataframe for returns 
    interval = int(len(y_test)/len(symbol_list))
    
    RtnPred_timeindex = plot_time_series_idx[range(split_idx, each_data_length-1+pred_len, pred_len)]
        
    stg_dict = dict()
    for i in range(len(symbol_list)):
        symbol = symbol_list[i]
        RtnPred_df = pd.DataFrame(y_pred[interval*i:interval*(i+1)], index = RtnPred_timeindex)
        RtnReal_df = returns[symbol][list(range(split_idx, each_data_length-1+pred_len, pred_len))]
        stg_df = pd.concat([RtnReal_df, RtnPred_df], axis = 1)
        stg_df.columns = ['real', 'pred']
        stg_dict[symbol] = stg_df
    
    return train_test_interval, loss, valid_acc(y_test, y_pred), stg_dict
    
    






















