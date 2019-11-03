import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import Trainer
import os
from durbango import pickle_save
from pathlib import Path


def main(args):
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs)
    scaler = dataloader['scaler']
    print(args)
    best_model_save_path = os.path.join(args.save, 'best_model.pth')

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    engine = Trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.do_graph_conv,
                     args.addaptadj, adjinit)
    print("start training...",flush=True)
    metrics, train_time = [], []
    best_yet = 100
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            loss, mape, rmse = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)
            if args.n_iters is not None and iter >= args.n_iters:
                break
        train_time.append(time.time()-t1)
        total_time, valid_loss, valid_mape, valid_rmse = eval_(dataloader['val_loader'], device, engine)
        print(f'Epoch: {i:03d}')

        m = pd.Series(dict(train_loss=np.mean(train_loss),
                           train_mape=np.mean(train_mape),
                           train_rmse=np.mean(train_rmse),
                           valid_loss=np.mean(valid_loss),
                           valid_mape=np.mean(valid_mape),
                           valid_rmse=np.mean(valid_rmse), ))
        metrics.append(m)
        print(f'Epoch {i}')
        print(m.round(4))
        if m.valid_loss < best_yet:
            torch.save(engine.model.state_dict(), best_model_save_path)
            best_yet = m.valid_loss
        met_df = pd.DataFrame(metrics)
        met_df.to_csv(f'{args.save}/metrics.csv')
    print(f"Training finished. Best Valid Loss")
    print(met_df.loc[met_df.valid_loss.idxmin()].round(4))
    # Metrics on test data
    engine.model.load_state_dict(torch.load(best_model_save_path))

    realy = torch.Tensor(dataloader['y_test']).transpose(1,3)[:,0,:,:].to(device)
    outputs = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]

    test_met = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        test_met.append([x.item() for x in util.cheaper_metric(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t').round(3)
    test_met_df.to_csv(os.path.join(args.save, 'test_metrics.csv'))
    print(test_met_df.mean().round(3))

def eval_(ds, device, engine):
    """Run validation."""
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for (x, y) in ds.get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)
        metrics = engine.eval(testx, testy[:, 0, :, :])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    total_time = time.time() - s1
    return total_time, valid_loss, valid_mape, valid_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=100, help='')
    # parser.add_argument('--print_every', type=int, default=50, help='')
    # parser.add_argument('--seed',type=int,default=99,help='random seed')
    parser.add_argument('--save', type=str, default='experiment', help='save path')
    # parser.add_argument('--expid', default=1, help='experiment id')
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations')

    args = parser.parse_args()
    t1 = time.time()

    pickle_save(args, f'{args.save}/args.pkl')
    main(args)
    t2 = time.time()
    mins = (t2 - t1)/60
    print(f"Total time spent: {mins:.2f} seconds")
