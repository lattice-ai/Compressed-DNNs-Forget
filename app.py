# Imports
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

pie_1 = Image.open("assets/PIE_1.png")
pie_2 = Image.open("assets/PIE_2.png")
pie_3 = Image.open("assets/PIE_3.png")

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
st.markdown("""This interactive web app aims to: 
* provide a easily accessible way to help explore some of the biases introduced by Model Compression and Pruning Techniques such as Constant Sparsity based Low Magnitude Pruning. 
* Verify some of the claims of the paper [What Do Compressed Deep Neural Networks Forget?](https://arxiv.org/pdf/1911.05248.pdf) by Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin and Andrea Frome""")
st.markdown("---")
st.markdown("All the model weights can be found in the [Artifacts section](https://wandb.ai/sauravmaheshkar/exploring-bias-and-compression/artifacts?workspace=) of the Weights and Biases Project Page and the tflite models can be found in the release section of the [github repository](https://github.com/SauravMaheshkar/Compressed-DNNs-Forget) accompanying this app")
st.header("Key Takeaways")
st.warning("""

1. Top-line metrics such as top-1 or top-5 test-set accuracy hide critical details in the ways that pruning impacts model generalization. Certain parts of the data distribution are far more sensitive to varying the number of weights in a network, and bear the brunt cost of varying the weight representation.

2. The examples most impacted by pruning, which the authors term Pruning Identified Exemplars(PIEs), are more challenging for both models and humans to classify. Compression impairs model ability to predict accurately on the long-tail of less frequent instances

3. Compressed networks are far more brittle than non-compressed models to small changes in the distribution that humans are robust to. This sensitivity is amplified at higher levels of compression.

Results on CelebA showed that PIE over-indexes on protected attributes like gender and age, suggesting that compression may amplify exisiting algorithmic bias. For sensitive taks, the introduction of pruning may be at odds with fairness objectives to avoid disparate treatment of protected attributes and/or the need to guarantee a level of recall for certain classes.

""")


## Sidebar
st.sidebar.header("What do Compressed Neural Networks Forget?")
st.sidebar.markdown("---")
st.sidebar.markdown("If you liked this project and would like to read the code and see some of my other work, don't forget to ⭐the [repository](https://github.com/SauravMaheshkar/Compressed-DNNs-Forget) and follow [me](https://github.com/SauravMaheshkar).")


st.header("Interactive Demo")
st.info("NOTE: A set of predictions are generated using the population of models. The modal label , i.e. the class predicted most frequently by the pruned models population for a image is noted and if the modal label is different from the output of the non-pruned model then the modal is characterised as a Pruning Identified Exemplar (PIE).")
## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png"])
if uploaded_file is not None:
    image_pred = np.asarray(bytearray(uploaded_file.read()))
    image_pred = Image.fromarray(image_pred).convert('RGB')
    open_cv_image = np.array(image_pred) 
    image_pred = cv2.resize(open_cv_image, (178, 218))
    img =  np.expand_dims(image_pred, axis=0).astype(np.float32)

st.markdown("You'll get 5 outputs from each of the pruned models { 0.3, 0.5, 0.7, 0.9 } and the baseline (non-pruned) model. If the **modal label** is different from the Predictions of the non-pruned model, then the image can be identified as a Pruning Identified Exemplar")

if st.button("Get Predictions"):
    suggestion = get_predictions(input_image = img)
    for i in suggestion:
        st.write(i)

st.markdown("---")
st.title("Some Pruning Identified Exemplars")
st.image([pie_1, pie_2, pie_3])

st.markdown("---")
st.markdown("If you liked this project and would like to read the code and see some of my other work, don't forget to ⭐the [repository](https://github.com/SauravMaheshkar/Compressed-DNNs-Forget) and follow [me](https://github.com/SauravMaheshkar).")

