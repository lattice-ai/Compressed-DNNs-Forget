# Internal
from config.config import CFG
from model.ForgetModel import Model

# External
import tensorflow as tf

tflite_interpreter = tf.lite.Interpreter(model_path="tflite_models/fifty_pruned.tflite")

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
tflite_interpreter.allocate_tensors()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])

model = Model(CFG)
model.load_data()
x = next(iter(model.test_generator))

tflite_interpreter.set_tensor(input_details[0]['index'], x)
tflite_interpreter.invoke()
tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])