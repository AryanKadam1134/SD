import unittest
from src.models.stress_detector import StressDetector

class TestStressDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = StressDetector()
        self.detector.load_data('path/to/affectnet/dataset')

    def test_predict_stress(self):
        # Test with a sample image
        result = self.detector.predict_stress('path/to/sample/image.jpg')
        self.assertIn(result, ['stress', 'no_stress'])

    def test_load_data(self):
        # Test if data is loaded correctly
        train_data, val_data, test_data = self.detector.get_data()
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertGreater(len(test_data), 0)

if __name__ == '__main__':
    unittest.main()