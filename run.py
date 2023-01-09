# -*- coding: utf-8 -*-

from src.data.make_dataset import xlsx_to_csv, split_train_test
from src.features.build_features import build_feature, series_to_supervised
from src.models.train_model import train
from src.models.predict_model import predict
from src.metrics.metrics import create_table_metrics, mean_absolute_deviation
from src.visualization.visualize import scaling_comparison, scaling_histogram_comparison

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import pandas as pd
import numpy as np
import psutil
import tracemalloc
import matplotlib.pyplot as plt

INTER_PATH = "./data/interim/"

INPUT_FILE = "./data/raw/ventas_lamatta.xlsx"
INTER_FILE = "./data/interim/venta_diaria.csv"
PROCC_FILE = "./data/processed/train_supervisado.csv"
MODEL_FILE = "./models/lamatta.h5"
FIGUR_PATH = "./reports/figures/"

"""Cargamos nuestro Dataset

Prepara el dataset raw convirtiendolo a csv para poder manejarlo con
mayor facilidad y extrae las columnas necesarias.
"""
xlsx_to_csv(INPUT_FILE, sheet_name='venta_diaria', cols=['FECHA', 'VV'], output=INTER_FILE)

split_train_test(INTER_FILE,
                 train_range=["2017-01-02", "2020-12-31"], 
                 test_range=["2021-01-02", "2021-09-23"],
                 output_path=INTER_PATH)

""" Creamos los features extraer los datos que alimenta el modelo el 7 verlo en el colob

Crea el dataset con la estructura necesario para los input de nuestro modelo
de predicción. Y retorna el scaler usado en nuestros datos para posteriores
predicciones.
"""
PASOS = 7
scaler, data, scaled = build_feature(INTER_PATH + "inter_train.csv", PASOS, PROCC_FILE)

"""
Plot scaling info
"""
scaling_comparison(data, scaled, FIGUR_PATH + "scaling_comparison.png")
scaling_histogram_comparison(data, scaled, FIGUR_PATH + "scaling_histogram_comparison.png")

# Aqui se activa el rastreado de memoria para saber el consumo de memoria en el modelo (antes de correr el modelo)
tracemalloc.start()
current, peak = tracemalloc.get_traced_memory()
print(f"La cantidad de memoria usada actualmente es {current / 10**6}MB; el pico fue de {peak / 10**6}MB")
tracemalloc.stop()

# Se activa el INICIO DE USO CPU
print('CPU usado es: ', psutil.cpu_percent(4))

""" Entrenamiento 

Carga el modelo con la estructura necesaria para nuestros datos.
Realiza la predicción y retorna los dataset de entrenamiento, validación,
el modelo, y las métricas.
"""
EPOCHS = 70
x_train, y_train, x_val, y_val, history, model = train(PROCC_FILE, PASOS, EPOCHS, MODEL_FILE)

history.history['loss']
history.history['val_loss']
history.history['mse']
history.history['val_mse']
history.history['mae']
history.history['val_mae']
history.history['mape']
history.history['val_mape']

""" Prediccion 

Realiza la predicción cargando el modelo previamente guardado.
"""
results = predict(MODEL_FILE, x_val)
print(len(results))

""" Reports """
plt.scatter(range(len(y_val)), y_val,c='g')
plt.scatter(range(len(results)), results,c='r')
plt.title('validate')
plt.savefig(FIGUR_PATH + "validate.png")
plt.show()

plt.plot(history.history['loss'])
plt.title('loss')
plt.savefig(FIGUR_PATH + "loss.png")
plt.show()

plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.savefig(FIGUR_PATH + "validate_loss.png")
plt.show()

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Accuracy')
plt.savefig(FIGUR_PATH + "accuracy.png")
plt.show()

"""
Saca las metricas en una tabla por cada prediccion
"""
df_res = create_table_metrics(y_val, results)
df_res.to_csv('./reports/data/comparacion_metrics.csv')

"""
********************************************************************
"""

"""COMPARACION DE LOS DATOS REALES Y PREDICTIVOS"""
compara = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
compara.columns = ['real', 'prediccion']
inverted = scaler.inverse_transform(compara)

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real', 'prediccion']
compara2['diferencia'] = compara2['real'] - compara2['prediccion']
compara2.head()
compara2[['real',	'prediccion',	'diferencia']]
compara2.describe()

compara2[['real',	'prediccion',	'diferencia']]
compara2.describe()
compara2.to_csv('./reports/data/comparacion.csv')

compara2.plot()
plt.savefig(FIGUR_PATH + "comparacion.png")

"""
Predicción de test (2021)
"""
df = pd.read_csv(INTER_PATH + "inter_test.csv").set_index("FECHA")
df.index = pd.to_datetime(df.index)

print("*****************************************")
print("TEST")
print("*****************************************")
print(f"Fecha minima: {df.index.min()}")
print(f"Fecha maxima: {df.index.max()}")
"""## Preparamos los datos para Test para darle los pesos correspondientes"""

values = df['2021-08-20':'2021-09-23'].values.astype('float32')

# normalize features
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, PASOS, 1)
x_test = reframed[reframed.columns[:PASOS]].values
y_test = reframed[reframed.columns[PASOS]].values

x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)

"""PREDICCION"""
y_pred = model.predict(x_test)

"""
Saca las metricas en una tabla por cada prediccion
"""
df_res = create_table_metrics(y_test, y_pred)
df_res.to_csv('./reports/data/pronostico_metrics.csv')

"""METRICAS"""
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mad_real = mean_absolute_deviation(y_test)
mad_pred = mean_absolute_deviation(y_pred)
print("Metricas para primera semana: \n")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"MAD real: {mad_real}")
print(f"MAD pred: {mad_pred}")
print("\n\n")

"""GRAFICOS"""
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape((-1, 1)))
print("INVERSE TRANSFORM:\n")
print(y_pred_inv)
print("\n")

"""## Visualizamos el pronóstico"""
df_pronostico = pd.DataFrame({"real": y_test_inv.ravel(), "pronostico": y_pred_inv.ravel()})
df_pronostico.to_csv('./reports/data/pronostico.csv')

plot = df_pronostico.plot()
fig = plot.get_figure()
fig.savefig(FIGUR_PATH + "prediccion.png")