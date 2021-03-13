"""
Sample Script for Model Training using the Weights and Biases Client
"""
import wandb
from config.config import CFG
from forgetfuldnn.model.ForgetModel import Model

wandb.login(key="")
wandb.init(project="", entity="", sync_tensorboard=True)

model = Model(CFG)
model.load_data()
model.build()
model.load(weights="weights/baseline.h5")
model.predict()
