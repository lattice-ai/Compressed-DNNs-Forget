# Internal
from config.config import CFG
from forgetfuldnn.model.ForgetModel import Model

model = Model(CFG)
model.build()
model.load_pruned(weights="weights/model_ninety.h5", factor=0.9)
model.export_tflite()
