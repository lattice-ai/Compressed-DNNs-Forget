# External
import unittest
import tensorflow as tf

# Internal 
from dataloader.dataloader import DataLoader
from config.config import CFG
from utils.config import Config

class TestDataLoader(unittest.TestCase):
    
    def test_dataloader(self):
        """
        Test for importing Dataset using the dataloader class
        """
        configuration = Config.from_json(CFG)
        loader = DataLoader()
        train_generator, validation_generator, test_generator = loader.load_data(configuration)
        self.assertIsInstance(train_generator, tf.keras.preprocessing.image.DirectoryIterator)  
        self.assertIsInstance(validation_generator, tf.keras.preprocessing.image.DirectoryIterator)
        self.assertIsInstance(test_generator, tf.keras.preprocessing.image.DirectoryIterator)        

if __name__ == '__main__':
    unittest.main() 