from .base_model import BaseModel

import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input

class Model(tf.keras.Model):

    def __init__(self, config):
        super().__init__(config)
        self.base_model = tf.keras.applications.InceptionV3(weights = "../weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            include_top = False, 
            input_shape = self.config.model.input)
        self.model = None