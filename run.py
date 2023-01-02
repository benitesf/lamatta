# -*- coding: utf-8 -*-

from src.data.make_dataset import xlsx_to_csv
from src.features.build_features import build_feature, series_to_supervised
from src.models.train_model import train
from src.models.predict_model import predict

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import pandas as pd
import numpy as np
import psutil
import tracemalloc
import matplotlib.pyplot as plt

INPUT_FILE = "./data/raw/ventas_lamatta.xlsx"
INTER_FILE = "./data/interim/venta_diaria.csv"
PROCC_FILE = "./data/processed/data_supervisado.csv"
MODEL_FILE = "./models/lamatta.h5"
FIGUR_PATH = "./reports/figures/"

"""Cargamos nuestro Dataset

Prepara el dataset raw convirtiendolo a csv para poder manejarlo con
mayor facilidad y extrae las columnas necesarias.
"""
xlsx_to_csv(INPUT_FILE, sheet_name='venta_diaria', cols=['FECHA', 'VV'], output=INTER_FILE)

""" Creamos los features extraer los datos que alimenta el modelo el 7 verlo en el colob

Crea el dataset con la estructura necesario para los input de nuestro modelo
de predicción. Y retorna el scaler usado en nuestros datos para posteriores
predicciones.
"""
PASOS = 7
scaler = build_feature(INTER_FILE, PASOS, PROCC_FILE)

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
EPOCHS = 10
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
print( len(results) )

""" Reports """
plt.scatter(range(len(y_val)), y_val,c='g')
plt.scatter(range(len(results)), results,c='r')
plt.title('validate')
plt.savefig(FIGUR_PATH + "validate.png")

plt.plot(history.history['loss'])
plt.title('loss')
plt.savefig(FIGUR_PATH + "loss.png")

plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.savefig(FIGUR_PATH + "validate_loss.png")

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Accuracy')
plt.savefig(FIGUR_PATH + "accuracy.png")

"""
********************************************************************
"""

"""COMPARACION DE LOS DATOS REALES Y PREDICTIVOS"""
compara = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
compara.columns = ['real', 'prediccion']
inverted = scaler.inverse_transform(compara.values)
inverted
compara.values

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real', 'prediccion']
compara2['diferencia'] = compara2['real'] - compara2['prediccion']
compara2.head()
compara2[['real',	'prediccion',	'diferencia']]
compara2.describe()

compara2[['real',	'prediccion',	'diferencia']]
compara2.describe()

plt.plot(compara2['real'])
plt.plot(compara2['prediccion'])
plt.title("Comparacion")
plt.savefig(FIGUR_PATH + "comparacion.png")

"""Predicción de los 23 días"""
df = pd.read_csv(INTER_FILE).set_index("FECHA")
df.index = pd.to_datetime(df.index)

ultimosDias = df['2021-09-01':'2021-09-23']
ultimosDias

"""## Preparamos los datos para Test para darle los pesos correspondientes"""

values = ultimosDias.values
values = values.astype('float32')
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
scaled

reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
print(reframed)

values = reframed.values
x_test = values[6:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)

"""Creamos una función que nos permita agregar los nuevos valores """

def agregarNuevoValor(x_test, nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test

"""## Pronóstico para la "próxima semana"
"""

results=[]
for i in range(7):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    x_test=agregarNuevoValor(x_test,parcial[0])

"""## Re-Convertimos los resultados"""

adimen = [x for x in results]
print("ADIMEN: \n")
print(adimen)
print("\n")

inverted = scaler.inverse_transform(adimen)
print("INVERSE TRANSFORM:\n")
print(inverted)
print("\n")

"""## Visualizamos el pronóstico"""
prediccion1Semana = pd.DataFrame(inverted)
prediccion1Semana.columns = ['pronostico']
prediccion1Semana.to_csv('./reports/data/pronostico.csv')

print("PREDICCION SEMANA:\n")
print(prediccion1Semana)
print("\n")


plot = prediccion1Semana.plot()
fig = plot.get_figure()
fig.savefig(FIGUR_PATH + "prediccion.png")


y_pred = inverted.reshape(inverted.shape[0])
y_real = ultimosDias.values[0:7, :]
y_real = y_real.reshape(y_real.shape[0])
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)
mape = mean_absolute_percentage_error(y_real, y_pred)
print("Metricas para primera semana: \n")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print("\n\n")

"""Saca los valores reales y sus predicciones"""
p_real = plt.scatter(range(len(y_real)), y_real,c='g')
p_pred = plt.scatter(range(len(y_pred)), y_pred,c='r')
plt.title('prediccion_7_dias')
plt.legend((p_real, p_pred),
           ("real", "prediccion"),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=8)
plt.savefig(FIGUR_PATH + "prediccion_7_dias.png")


"""# Agregamos el resultado en el dataset"""

i=0
for fila in prediccion1Semana.pronostico:
    i=i+1
    ultimosDias.loc['2021-00-0' + str(i) + ' 00:00:00'] = fila
    print(fila)
ultimosDias.tail(14)