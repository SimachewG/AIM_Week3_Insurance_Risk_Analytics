import unittest
import pandas as pd
from src.exploratory_data_analysis import compute_loss_ratio
from src.data_preprocessing import load_data, preprocess_data

class TestExploratoryDataAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = pd.load_data('../data/raw/MachineLearningRating_v3.txt')
        self.data = preprocess_data(self.data)

    def test_compute_loss_ratio(self):
        loss_ratio = compute_loss_ratio(self.data)
        self.assertGreaterEqual(loss_ratio, 0)  # Loss ratio should be non-negative

if __name__ == '__main__':
    unittest.main()