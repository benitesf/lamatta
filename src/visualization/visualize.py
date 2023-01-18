# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

REPORT_DATA = "./reports/data/"
FIGURE_PATH = "./reports/figures/"

def scaling_comparison(data, scaled, outpath):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5))    
    ax[0].plot(data, color="blue")
    ax[1].plot(scaled, color="green")
    fig.savefig(outpath)
    plt.show()
    
def scaling_histogram_comparison(data, scaled, outpath):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5), tight_layout=True)    
    ax[0].hist(data, bins=20, color="blue")
    ax[1].hist(scaled, bins=20, color="green")
    fig.savefig(outpath)
    plt.show()

def true_and_prediction(y_true, y_pred, scaler, filename):
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y_true.reshape((-1, 1)))

    df_pronostico = pd.DataFrame({"real": y_true_inv.ravel(), "pronostico": y_pred_inv.ravel()})
    df_pronostico.to_csv(REPORT_DATA + filename + ".csv")

    plot = df_pronostico.plot()
    fig = plot.get_figure()
    plt.title("Prediccion TEST")
    fig.savefig(FIGURE_PATH + filename +".png")

def plot_accumulative_prediction(data, ndays):
    b_patch = mpatches.Patch(color="blue", label="Datos reales")
    o_patch = mpatches.Patch(color="orange", label="Datos predictivos")

    y = list(range(len(data)))
    fig, ax = plt.subplots()
    ax.plot(y[:len(y) - ndays + 1], data[:len(data) - ndays + 1], color="blue")
    ax.plot(y[ndays:], data[ndays:], color="orange")
    plt.legend(handles=[b_patch, o_patch])
    plt.title(f"Predicción {ndays} días")
    fig.savefig(FIGURE_PATH + f"prediccion_{ndays}.png")
    plt.show()
    
    df = pd.DataFrame({"prediccion": data.ravel()})
    df.to_csv(REPORT_DATA + f"prediccion_{ndays}_days.csv", index=False)
