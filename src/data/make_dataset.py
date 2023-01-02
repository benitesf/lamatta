# -*- coding: utf-8 -*-
import pandas as pd

REPORT_DATA_PATH = "./reports/data/"

def xlsx_to_csv(filepath, sheet_name, cols, output):
    """Lee un fichero xlsx, procesa y almacena en csv"""
    df = pd.read_excel(filepath, sheet_name=sheet_name, usecols=cols)
    # Calculamos el total Venta por dia (Acumulados)
    df = df.groupby('FECHA').sum()
    print("\nDataFrame INFO:\n")
    print(df.info())
    print("\n")
    df.to_csv(output)