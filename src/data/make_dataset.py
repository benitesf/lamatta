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
    
def split_train_test(filepath, train_range, test_range, output_path):
    df = pd.read_csv(filepath).set_index("FECHA")
    df.index = pd.to_datetime(df.index)
    df.index.sort_values()
    
    print(f"Train range: {train_range[0]} to {train_range[1]}")
    train_df = df[train_range[0]:train_range[1]]
    
    print(f"Test range: {test_range[0]} to {test_range[1]}")
    test_df = df[test_range[0]:test_range[1]]
    
    train_df.to_csv(output_path + "inter_train.csv")
    test_df.to_csv(output_path + "inter_test.csv")
    