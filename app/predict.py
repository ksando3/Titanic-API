import joblib
import numpy as np
import pandas as pd

def load_model(path = "model.pkl"):
    model = joblib.load(path)
    return model

def predict(model, data):
    df = pd.DataFrame(data)
    return model.predict(df)
