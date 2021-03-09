"""
Sample Script for Model Training
"""
from config.config import CFG
from model.ForgetModel import Model

model = Model(CFG)
model.load_data()
model.build()
model.load(weights="weights/baseline.h5")
model.predict()

