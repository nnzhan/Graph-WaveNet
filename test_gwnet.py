from train import main
import test
import unittest
from durbango import pickle_load
import pandas as pd
import os
import shutil

TRAIN_ARGS = 'test_args.pkl'
TEST_ARGS = 'test_script_args.pkl'
SAVE_DIR = 'utest_experiment/'

class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR)

    def test_1_epoch(self):
        args = pickle_load(TRAIN_ARGS)
        args.epochs = 2
        args.n_iters = 1
        args.batch_size = 2
        args.n_obs = 2
        args.save = SAVE_DIR
        main(args)
        df = pd.read_csv(f'{args.save}/metrics.csv', index_col=0)
        self.assertEqual(df.shape, (2,6))
        test_df = pd.read_csv(f'{args.save}/test_metrics.csv', index_col=0)
        self.assertEqual(test_df.shape, (12, 3))

    def test_test_script(self):
        train_args = pickle_load(TRAIN_ARGS)
        test_args = pickle_load('test_script_args.pkl')
        test_args.checkpoint = train_args.save + '/best_model.pth'
        test_args.n_obs = 2
        test.main(test_args)

