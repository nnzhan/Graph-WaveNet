from train import main
import test
import unittest
from durbango import pickle_load
import pandas as pd
import os
import torch
import util
import shutil

TRAIN_ARGS = 'test_args.pkl'
TEST_ARGS = 'test_script_args.pkl'
SAVE_DIR = 'utest_experiment/'
ARG_UPDATES = {'epochs': 1, 'n_iters': 1, 'batch_size': 2, 'n_obs': 2,
               'device': util.DEFAULT_DEVICE, 'save': SAVE_DIR, 'addaptadj': True,
               'apt_size': 2, 'nhid': 1, 'lr_decay_rate': 1.,
               'in_dim': 1, 'cat_feat_gc': True, 'clip': 1, 'es_patience': 10,
               'checkpoint': '', 'fill_zeroes': False}

MODEL_KWARGS = {'end_channels': 4, 'skip_channels': 2}
def modify_args(args, updates):
    for k,v in updates.items():
        setattr(args, k, v)
    return args

class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR)

    def test_1_epoch(self):
        args = modify_args(pickle_load(TRAIN_ARGS), ARG_UPDATES)
        args.fp16 = ''
        engine = main(args, **MODEL_KWARGS)
        df = pd.read_csv(f'{args.save}/metrics.csv', index_col=0)
        self.assertEqual(df.shape, (args.epochs, 6))
        test_df = pd.read_csv(f'{args.save}/test_metrics.csv', index_col=0)
        self.assertEqual(test_df.shape, (12, 3))
        test_args = modify_args(pickle_load(TEST_ARGS), ARG_UPDATES)
        test_args.checkpoint = SAVE_DIR + '/best_model.pth'
        state_dict = torch.load(test_args.checkpoint)
        self.assertTrue('nodevec1' in state_dict)
        self.assertTrue(os.path.exists(test_args.checkpoint))
        new_met, new_preds = test.main(test_args, **MODEL_KWARGS)
        deltas = test_df.mean() - new_met.mean()
        self.assertGreaterEqual(.01, deltas.abs().max())
