from train import main
import unittest
from durbango import pickle_load
import pandas as pd
TEST_ARGS_PATH = 'test_args.pkl'

class TestTrain(unittest.TestCase):

    def test_1_epoch(self):
        args = pickle_load(TEST_ARGS_PATH)
        args.epochs = 2
        args.n_iters = 1
        args.batch_size = 4
        args.n_obs = 4
        main(args)
        df = pd.read_csv(f'{args.save}/metrics.csv', index_col=0)
        self.assertEqual(df.shape, (2,6))
        test_df = pd.read_csv(f'{args.save}/test_metrics.csv', index_col=0)
        self.assertEqual(test_df.shape, (12, 3))
