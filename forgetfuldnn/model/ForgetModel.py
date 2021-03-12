# Internal
from .base_model import BaseModel
from forgetfuldnn.dataloader.dataloader import DataLoader

# External
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot


class Model(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.img_height = int(self.config.data.IMG_HEIGHT)
        self.img_width = int(self.config.data.IMG_WIDTH)
        self.base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(self.img_height, self.img_width, 3))

        self.model = None
        self.pruned_model= None
        self.training_samples = int(self.config.data.TRAINING_SAMPLES)
        self.batch_size = int(self.config.train.BATCH_SIZE)
        self.steps_per_epoch = int(self.training_samples) // int(self.batch_size)
        self.num_epochs = int(self.config.train.EPOCHS)

        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

        self.predictions = None

    def load_data(self):
        """
        Loads Training and Validation Generator from the DataLoader Class
        """
        self.train_generator, self.validation_generator, self.test_generator = DataLoader().load_data(self.config)

    def build(self):
        """
        Build a Keras Model from the InceptionV3 backbone
        """
        x = self.base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)

        self.model = tf.keras.models.Model(inputs = self.base_model.input, outputs = predictions)

        for layer in self.model.layers[:52]:
            layer.trainable = False

        # return self.model

    def train(self):
        """
        Abstract Method to Train the Model and Return the Training Loss and Validation Loss
        """
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), 
               loss='categorical_crossentropy', 
               metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k = 1)])

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.best.inc.blond.hdf5', 
                               verbose=1, save_best_only=True)

        model_history = self.model.fit(self.train_generator,
                                        validation_data = self.validation_generator,
                                        steps_per_epoch= self.steps_per_epoch,
                                        epochs = self.num_epochs,
                                        callbacks=[checkpointer])

        self.model.save("baseline.h5")

        return model_history.history['loss'], model_history.history['val_loss']

    def load(self, weights):
        
        self.model.load_weights(weights)

    def prune(self, factor):
        
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(factor, 0),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(self.model,**pruning_params)


        log_dir_thirty = tempfile.mkdtemp()

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir_thirty),
        ]

        self.pruned_model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), 
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k = 1)])

        self.pruned_model.fit(self.train_generator, 
                            validation_data = self.validation_generator,
                            callbacks = callbacks,
                            steps_per_epoch= self.steps_per_epoch,
                            epochs = self.num_epochs)
                            
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        print('Saved pruned Keras model to:', pruned_keras_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        pruned_tflite_model = converter.convert()

        _, pruned_tflite_file = tempfile.mkstemp('.tflite')

        with open(pruned_tflite_file, 'wb') as f:
            f.write(pruned_tflite_model)

        print('Saved pruned TFLite model to:', pruned_tflite_file)

    def predict(self):

        self.predictions = self.model.predict(self.test_generator)