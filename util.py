import argparse
import pickle
import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=False):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, n_obs=None, fill_zeroes=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std(), fill_zeroes=fill_zeroes)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds-labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, device, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def make_graph_inputs(args, device):
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports


def get_shared_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=40, help='Number of channels for internal conv')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--apt_size', default=10, type=int)
    parser.add_argument('--cat_feat_gc', action='store_true')
    parser.add_argument('--fill_zeroes', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='')
    return parser
