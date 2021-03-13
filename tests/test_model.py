# External
import unittest
import tensorflow as tf

# Internal
from config.config import CFG
from forgetfuldnn.model.ForgetModel import Model


class TestModel(unittest.TestCase):
    """
    Custom Class for Checking each Abstract Method in our Model Class
    """

    def test_model(self):
        """
        Test for Instantiating a Model using the Model class
        """
        # configuration = Config.from_json(CFG)
        model = Model(CFG)
        model.build()
        self.assertIsInstance(model.model, tf.keras.models.Model)


if __name__ == "__main__":
    unittest.main()
