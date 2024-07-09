# -*- coding: utf-8 -*-

from src.features.build_features import series_to_supervised

import keras
import numpy as np

import psutil
import tracemalloc
import time

def predict(model_file, x_val):
    model = keras.models.load_model(model_file)
    return model.predict(x_val)

def predict_from_date(model_file, scaler, filepath):
    pass

def predict_n_days(model, scaler, X, pasos, ndays):    
    X = scaler.transform(X)
    acc = [i for i in X.ravel()]    
    X = X.reshape((1, 1, pasos))

    y_pred = model.predict(X).ravel()[0]
    n_pred = [y_pred]
    acc.append(y_pred)

    for i in range(1, ndays):
        X = np.array(acc[len(acc)-pasos:])
        X = X.reshape((1, 1, pasos))
        y_pred = model.predict(X).ravel()[0]
        n_pred.append(y_pred)
        acc.append(y_pred)
        
    n_pred = scaler.inverse_transform(np.array(n_pred).reshape(-1, 1))
    acc = scaler.inverse_transform(np.array(acc).reshape(-1, 1))

    return n_pred, acc

    

def predict_test(model, scaler, data, pasos):
    # normalize features
    scaled = scaler.transform(data)

    reframed = series_to_supervised(scaled, pasos, 1)
    x = reframed[['var1(t-7)', 'var1(t-6)', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)','var1(t-2)', 'var1(t-1)']].values
    y = reframed['var1(t)'].values

    x = x.reshape((x.shape[0], 1, x.shape[1]))

    """PREDICCION"""
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    t_0 = time.time()
    
    y_pred = model.predict(x)
    
    t_1 = time.time()

    """se finaliza el uso de CPU en el entrenamiento"""
    print("\nMétricas de TESTING:\n")
    print('CPU usado es: ', psutil.cpu_percent(4))

    """Se finaliza el proceso de uso de memoria en el entrenamiento"""
    current, peak = tracemalloc.get_traced_memory()

    print(f"La cantidad de memoria usada actualmente es {current / 10**6}MB; el pico fue de {peak / 10**6}MB\n\n")
    tracemalloc.stop()    

    return y, y_pred 
