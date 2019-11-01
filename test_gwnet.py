from train import main
import unittest
from durbango import pickle_load
TEST_ARGS_PATH = 'test_args.pkl'

class TestTrain(unittest.TestCase):

    def test_1_epoch(self):
        args = pickle_load(TEST_ARGS_PATH)
        args.epochs = 1
        args.n_iters = 1
        args.batch_size = 4
        args.n_obs = 4
        main(args)
