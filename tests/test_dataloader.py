# External
import unittest
import keras

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
        train_generator = loader.get_train_data(configuration)
        self.assertIsInstance(train_generator, keras.preprocessing.image.NumpyArrayIterator)       

if __name__ == '__main__':
    unittest.main() 