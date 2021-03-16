# Imports
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt

def get_predictions(input_image):
    tflite_model_predictions = []
    for i in os.listdir("tflite_models/"):
        tflite_interpreter = tf.lite.Interpreter(model_path="tflite_models/" + str(i))
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.set_tensor(input_details[0]["index"], input_image)
        tflite_interpreter.invoke()
        tflite_model_predictions.append(tflite_interpreter.get_tensor(output_details[0]["index"]))
    return tflite_model_predictions

## Page Title
st.set_page_config(page_title = "What do Compressed Neural Networks Forget?", page_icon = "🧐")
st.title("What do Compressed Neural Networks Forget?")
st.markdown("---")

## Sidebar
st.sidebar.header("What do Compressed Neural Networks Forget?")
st.sidebar.markdown("---")
st.sidebar.markdown("")

## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png"])
if uploaded_file is not None:
    image_pred = np.asarray(bytearray(uploaded_file.read()))
    image_pred = Image.fromarray(image_pred).convert('RGB')
    open_cv_image = np.array(image_pred) 
    image_pred = cv2.resize(open_cv_image, (178, 218))
    img =  np.expand_dims(image_pred, axis=0).astype(np.float32)

if st.button("Predict"):
    suggestion = get_predictions(input_image = img)
    for i in suggestion:
        st.write(i)
