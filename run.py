# -*- coding: utf-8 -*-

from src.data.make_dataset import xlsx_to_csv_2, split_train_test
from src.features.build_features import build_feature
from src.models.train_model import train
from src.models.predict_model import predict, predict_test, predict_n_days
from src.metrics.metrics import create_table_metrics, calculate_metrics
from src.visualization.visualize import scaling_comparison, scaling_histogram_comparison, true_and_prediction, plot_accumulative_prediction
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import psutil
import tracemalloc
import matplotlib.pyplot as plt
import datetime
import sys

INTER_PATH = "./data/interim/"

INPUT_FILE = "./data/raw/venta_mensual.xlsx"
INTER_FILE = "./data/interim/venta_mensual.csv"
PROCC_FILE = "./data/processed/venta_mensual_train.csv"
MODEL_FILE = "./models/venta_mensual.h5"
FIGUR_PATH = "./reports/figures/"

"""Cargamos nuestro Dataset

Prepara el dataset raw convirtiendolo a csv para poder manejarlo con
mayor facilidad y extrae las columnas necesarias.
"""
xlsx_to_csv_2(INPUT_FILE,
              sheet_name='Hoja2',
              cols=['AÑO', 'MES', 'PRODUCTO', 'CANTIDAD PRODUCIDA'],
              output=INTER_FILE)

split_train_test(INTER_FILE,
                 train_range=["2017-01-01", "2022-05-01"],
                 test_range=["2022-06-01", "2024-04-01"],
                 output_path=INTER_PATH)

""" Creamos los features

Crea el dataset con la estructura necesario para los input de nuestro modelo
de predicción. Y retorna el scaler usado en nuestros datos para posteriores
predicciones.
"""
PASOS = 7
scaler, data, scaled = build_feature(INTER_PATH + "inter_train.csv",
                                     PASOS,
                                     'VENTAS',
                                     PROCC_FILE)

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
x_train, y_train, x_val, y_val, history, model = train(PROCC_FILE,
                                                       PASOS,
                                                       EPOCHS,
                                                       MODEL_FILE)

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
y_plt = plt.scatter(range(len(y_val)), y_val, marker='o', c='g')
r_plt =plt.scatter(range(len(results)), results, marker='x', c='r')
plt.legend((y_plt, r_plt),
           ('real', 'prediccion'),
           scatterpoints=1,
           loc='upper right',
           ncol=1)
plt.title('validate')
plt.savefig(FIGUR_PATH + "validate.png")
# plt.show()

plt.plot(history.history['loss'])
plt.title('loss')
plt.savefig(FIGUR_PATH + "loss.png")
# plt.show()

plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.savefig(FIGUR_PATH + "validate_loss.png")
# plt.show()

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

compara2.plot(title="Conjunto de Validacion")
plt.savefig(FIGUR_PATH + "comparacion.png")

"""
********************************************************************
"""

"""
Predicción de test (2023-2024)
"""
df = pd.read_csv(INTER_PATH + "inter_test.csv").set_index("FECHA")
df.index = pd.to_datetime(df.index)
df = df[df['VENTAS'].str.isnumeric()]

start = df.index.min()
end = df.index.max()

print("*****************************************")
print("TEST")
print("*****************************************")
print(f"Fecha minima: {start}")
print(f"Fecha maxima: {end}")
"""## Preparamos los datos para Test para darle los pesos correspondientes"""

"""
Predicción y gráficos del conjuntos de datos TESTING
"""
print(f"\nTesting de {start} a {end}")
# values = df[start:end].values.astype('float32')
values = df.values.astype('float32')
y_true, y_pred = predict_test(model, scaler, values, PASOS)

"""
Saca las metricas en una tabla por cada prediccion
"""
df_res = create_table_metrics(y_true, y_pred)
df_res.to_csv('./reports/data/pronostico_metrics.csv')

"""METRICAS"""
calculate_metrics(y_true, y_pred)

"""
CSV y grafico de la predicción
"""
true_and_prediction(y_true, y_pred, scaler, "prediccion")

"""
********************************************************************
"""
"""
Predicción acumulativa de 4 meses
"""
print("*****************************************")
print("Predicción 4 meses")
print("*****************************************")
N = 4

e_dt = datetime.datetime(2023, 12, 1, 0, 0, 0)
s_dt = e_dt - relativedelta(months=PASOS-1)

# start = np.datetime_as_string(s_dt)
# end = np.datetime_as_string(e_dt)

values = df[s_dt:e_dt].values.astype('float32')
n_pred, acc = predict_n_days(model, scaler, values, PASOS, N)

plot_accumulative_prediction(acc, N)
print(n_pred)