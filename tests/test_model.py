# External
import unittest
import keras

# Internal 
from config.config import CFG
from model.ForgetModel import Model

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
        self.assertIsInstance(model.build(), keras.models.Model)  

    def test_data(self):
        """
        Test for Dataloaders created using the load_data method
        """

        model = Model(CFG)
        train_generator, validation_generator = model.load_data()
        self.assertIsInstance(train_generator, keras.preprocessing.image.DirectoryIterator)  
        self.assertIsInstance(validation_generator, keras.preprocessing.image.DirectoryIterator)  

if __name__ == '__main__':
    unittest.main() 