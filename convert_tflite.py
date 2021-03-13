# Internal
from config.config import CFG
from forgetfuldnn.model.ForgetModel import Model

# External
import tensorflow as tf

model = Model(CFG)
model.build()
model.load_pruned(weights="weights/model_seventy.h5", factor=0.7)
model.export_tflite()
