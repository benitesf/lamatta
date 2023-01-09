# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def create_table_metrics(y_true, y_pred):
    
    data = np.column_stack((y_true.ravel(), y_pred.ravel()))
    err = []                           
    mse = []
    mae = []
    mape = []
    
    for i in range(0, y_true.shape[0]):
        err.append(data[i:i+1, 0] - data[i:i+1, 1])
        mse.append(mean_squared_error(data[i:i+1, 0], data[i:i+1, 1]))
        mae.append(mean_absolute_error(data[i:i+1, 0], data[i:i+1, 1]))
        mape.append(mean_absolute_percentage_error(data[i:i+1, 0], data[i:i+1, 1]))
    
    data = np.column_stack((data, err, mse, mae, mape))
    df = pd.DataFrame(data)
    df.columns = ["real", "pred", "err", "mse", "mae", "mape"]
    
    return df

def mean_absolute_deviation(data):
    return np.average(np.abs(data - data.mean()))