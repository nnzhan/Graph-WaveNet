import torch
import numpy as np
import pandas as pd
import time
import util
from engine import Trainer
import os
from durbango import pickle_save
from fastprogress import progress_bar

from model import GWNet
from util import calc_tstep_metrics
from exp_results import summary

bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']


def surgery(model, surg_checkpoint):
    state_dict = torch.load(surg_checkpoint)
    b,w = state_dict.pop(bk), state_dict.pop(wk)
    model.load_state_dict(state_dict, strict=False)
    cur_state_dict = model.state_dict()
    cur_state_dict[bk][:b.shape[0]] = b
    cur_state_dict[wk][:w.shape[0]] = w
    model.load_state_dict(cur_state_dict)
    return model



def main(args, **model_kwargs):
    log_test = getattr(args, 'log_test', False)
    device = torch.device(args.device)
    data = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs)
    scaler = data['scaler']
    aptinit, supports = util.make_graph_inputs(args, device)

    model = GWNet.from_args(args, device, supports, aptinit, **model_kwargs)
    if args.checkpoint:
        model = surgery(model, args.checkpoint)
    model.to(device)
    engine = Trainer.from_args(model, scaler, args)
    metrics = []
    best_model_save_path = os.path.join(args.save, 'best_model.pth')
    lowest_mae_yet = 100  # high value, will get overwritten
    mb = progress_bar(list(range(1, args.epochs + 1)))
    since_best = 0
    for _ in mb:
        train_loss, train_mape, train_rmse = [], [], []
        data['train_loader'].shuffle()
        for iter, (x, y) in enumerate(data['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            yspeed = trainy[:, 0, :, :]
            if yspeed.max() == 0: continue
            mae, mape, rmse = engine.train(trainx, yspeed)
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            if args.n_iters is not None and iter >= args.n_iters:
                break
        engine.scheduler.step()
        _, valid_loss, valid_mape, valid_rmse = eval_(data['val_loader'], device, engine)
        m = dict(train_loss=np.mean(train_loss), train_mape=np.mean(train_mape),
                 train_rmse=np.mean(train_rmse), valid_loss=np.mean(valid_loss),
                 valid_mape=np.mean(valid_mape), valid_rmse=np.mean(valid_rmse))
        if log_test:
            _, test_loss, test_mape, test_rmse = eval_(data['test_loader'], device, engine)
            mn = np.mean
            test_m = dict(test_loss=mn(test_loss), test_mape=mn(test_mape), test_rmse=mn(test_rmse))
            m.update(test_m)

        m = pd.Series(m)
        metrics.append(m)
        if m.valid_loss < lowest_mae_yet:
            torch.save(engine.model.state_dict(), best_model_save_path)
            lowest_mae_yet = m.valid_loss
            since_best = 0
        else:
            since_best += 1
        met_df = pd.DataFrame(metrics)
        mb.comment = f'best val_loss: {met_df.valid_loss.min(): .3f}, current val_loss: {m.valid_loss:.3f}, current train loss: {m.train_loss: .3f}'
        met_df.round(6).to_csv(f'{args.save}/metrics.csv')
        if since_best >= args.es_patience: break  #
    # Metrics on test data
    engine.model.load_state_dict(torch.load(best_model_save_path))
    realy = torch.Tensor(data['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
    test_met_df, yhat = calc_tstep_metrics(engine.model, device, data['test_loader'], scaler, realy, args.seq_length)
    test_met_df.round(6).to_csv(os.path.join(args.save, 'test_metrics.csv'))
    print(summary(args.save))

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
    parser = util.get_shared_arg_parser()
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--clip', type=int, default=5, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument('--save', type=str, default='experiment', help='save path')
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')
    parser.add_argument('--end_conv_lr', type=float, default=None)

    args = parser.parse_args()
    t1 = time.time()
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    pickle_save(args, f'{args.save}/args.pkl')
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} seconds")
