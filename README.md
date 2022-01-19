# LSTM model on predicting cryptocurrency
Exploration of predicting cryptocurrency's up and down on the next period using LSTM model using Python.

requirements.txt: list of required packages.

model_on_different_datasets.py (main file):
           This file calls LSTM.py in LSTM_categorical (LSTM model results). 
           Compare predicted return and actual return to generate return plot, and shows the trade counts according to our prediction.
           We can test different parameters and test on different time periods here.

LSTM.py: 
           This file includes the entire procedure that generates one LSTM prediction result, which calls all other auxiliary files. 
           If we want to change model details or what factors to include, revise here.

data_for_model.py: split data into train and test. Please read split_time function carefully to make a clean cut,
           since we introduced "window" into data, we want to avoid data leakage.

clean_data.py: The data I had is already clean. But in case some data are not. Clean it here.

factors_genertor: (The actual file factors_helper.py that creates factors are kept private) geneartor factors.
           And for empty values introduced during procedure of creating factors, I used forward fill-in.
           
genetic_programming.py: GP procedure. 

IC_analysis.py: analyzing correlation between each factor and the return of the next period, also excluded highly-correlated factors.
           Ranked factor ICs to check the validity of our factors.
           
Normalization.py: normalize data and added boundaries to reduce impact of outliers.

If you have any question, feel free to let me know. Best wishes.
