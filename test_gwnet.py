from train import main
import unittest
from durbango import pickle_load
TEST_ARGS_PATH = 'test_args.pkl'

class TestTrain(unittest.TestCase):

    def test_1_epoch(self):
        args = pickle_load(TEST_ARGS_PATH)
        args.epochs = 1
        main(args)
