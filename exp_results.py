import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt


def summary(d):
    try:
        tr_val = pd.read_csv(f'{d}/metrics.csv', index_col=0)
        tr_ser = tr_val.loc[tr_val.valid_loss.idxmin()]
        tr_ser['best_epoch'] = tr_val.valid_loss.idxmin()
        tr_ser['min_train_loss'] = tr_val.train_loss.min()
    except FileNotFoundError:
        tr_ser = pd.Series()
    try:
        tmet = pd.read_csv(f'{d}/test_metrics.csv', index_col=0)
        tmean = tmet.add_prefix('test_').mean()

    except FileNotFoundError:
        tmean = pd.Series()
    tab = pd.concat([tr_ser, tmean]).round(3)
    return tab

def loss_curve(d):
    if 'logs' not in d: d =  f'logs/{d}'
    tr_val = pd.read_csv(f'{d}/metrics.csv', index_col=0)
    return tr_val[['train_loss', 'valid_loss']]


def plot_loss_curve(log_dir):
    d = loss_curve(log_dir)
    ax = d.plot()
    plt.axhline(d.valid_loss.min())
    print(d.valid_loss.idxmin())

def make_table():
    return pd.DataFrame({os.path.basename(c): summary(c) for c in glob('logs/*')}).T.sort_values('valid_loss')
