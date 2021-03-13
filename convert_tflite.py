# Internal
from config.config import CFG
from forgetfuldnn.model.ForgetModel import Model

model = Model(CFG)
model.build()
model.load_pruned(weights="", factor=None)
model.export_tflite()
