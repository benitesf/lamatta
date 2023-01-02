# -*- coding: utf-8 -*-

import keras

def predict(model_file, x_val):
    model = keras.models.load_model(model_file)
    return model.predict(x_val)

def predict_from_date(model_file, scaler, filepath):
    pass