# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

REPORT_DATA_PATH = "./reports/data/"

def build_feature(input, pasos, colname, output):
    df = pd.read_csv(input, index_col=False).set_index("FECHA")
    df.index = pd.to_datetime(df.index)
    
    # Creamos un indice y asignamos como indice al campo Fecha, por que trabajaremos con serie de tiempo
    # df = df.resample('D').mean()
    # df = df.rolling(2).mean()
    
    # Verificar valores nulos
    # df.loc[df[colname].isnull()]
    
    # Detectar valores extraños
    # df[df[colname].isin([np.nan, np.inf, -np.inf])]
    
    # Reemplazar valores NaN
    # df = df[df[colname].str.isnumeric()]
    df[colname] = df[colname].fillna(0)
    df = df.drop(df[df[colname]==0].index)

    # load dataset
    values = df.values
    
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values = values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, pasos, 1)
    reframed.head().to_csv(REPORT_DATA_PATH + "feature_head.csv")
    reframed.describe().to_csv(REPORT_DATA_PATH + "feature_describe.csv")
    reframed.to_csv(output, index=False)

    return scaler, values, scaled

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def build_features_from(input, scaler, pasos, start_date, end_date):
    df = pd.read_csv(input).set_index("FECHA")
    df.index = pd.to_datetime(df.index)

    ultimosDias = df[start_date:end_date]
    
    values = ultimosDias.values.astype('float32')
    # normalize features
    values = values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)

    reframed = series_to_supervised(scaled, pasos, 1)
    reframed.drop(reframed.columns[[7]], axis=1, inplace=True)

    values = reframed.values
    x_test = values[6:, :]
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return x_test