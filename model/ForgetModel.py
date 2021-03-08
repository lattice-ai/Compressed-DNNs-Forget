# Internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# External
import tensorflow as tf

class Model(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.img_height = int(self.config.data.IMG_HEIGHT)
        self.img_width = int(self.config.data.IMG_WIDTH)
        self.base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(self.img_height, self.img_width, 3))

        self.model = None
        self.training_samples = int(self.config.data.TRAINING_SAMPLES)
        self.batch_size = int(self.config.train.BATCH_SIZE)
        self.steps_per_epoch = int(self.training_samples) // int(self.batch_size)
        self.num_epochs = int(self.config.train.EPOCHS)

        self.train_generator = None
        self.validation_generator = None

    def load_data(self):
        """
        Loads Training and Validation Generator from the DataLoader Class
        """
        self.train_generator, self.validation_generator = DataLoader().load_data(self.config)

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
