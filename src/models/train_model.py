# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Flatten

import psutil
import pandas as pd
import tracemalloc
import time

def train(input, pasos, epochs, output):
    ## Dividimos en set de Entrenamiento y Validación
    # split into train and test 
    df = pd.read_csv(input)
    values = df.values
    
    sep = values.shape[0] - 15
    
    train = values[:sep, :]
    val = values[sep + 1:, :]
    
    # split into input and outputs
    x_train, y_train = train[:, :-1], train[:, -1]
    x_val, y_val = val[:, :-1], val[:, -1]
    
    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))    
    x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    """Se ejecutará nuestro modelo y mostraremos como se comporta la maquina despues de 40 épocas."""
    #aqui se activa el rastreado de memoria para saber el consumo de memoria en el modelo (Cuando empieza a correr el modelo)
    model = crear_modeloFF(pasos)
    
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    t_0 = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val,y_val),batch_size=pasos)
    t_1 = time.time()

    """se finaliza el uso de CPU en el entrenamiento"""
    print("Métricas de entrenamiento:")
    print('CPU usado es: ', psutil.cpu_percent(4))

    """Se finaliza el proceso de uso de memoria en el entrenamiento"""
    current, peak = tracemalloc.get_traced_memory()

    print(f"La cantidad de memoria usada actualmente es {current / 10**6}MB; el pico fue de {peak / 10**6}MB")
    tracemalloc.stop()

    """VISUALIZAMOS LOS RESULTAMOS DEL ENTRENAMIENTO"""
    #REsultado de datos del Entrenamiento
    """history.history['loss']    
    history.history['val_loss']
    history.history['mse']
    history.history['val_mse']
    history.history['mae']
    history.history['val_mae']
    history.history['mape']
    history.history['val_mape']
    """

    # Guardamos el modelo
    model.save(output)
    
    return x_train, y_train, x_val, y_val, history, model

"""# Creamos el Modelo de Red Neuronal
## Utilizaremos una Red "normal" Feedforward y el optimizador Adam y definimos los datos a obtener MSE,MAE,MAPE
"""
def crear_modeloFF(pasos):
    model = Sequential() 
    model.add(Dense(pasos, input_shape=(1,pasos), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=["mse","mae","mape"])
    model.summary()
    return model




