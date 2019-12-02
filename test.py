import util
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(args, **model_kwargs):
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    model = GWNet(device, args.num_nodes, args.dropout, supports=supports,
                  do_graph_conv=args.do_graph_conv, addaptadj=args.addaptadj, aptinit=adjinit,
                  in_dim=args.in_dim, apt_size=args.apt_size, out_dim=args.seq_length,
                  residual_channels=args.nhid, dilation_channels=args.nhid,
                  skip_channels=args.nhid * 8, end_channels=args.nhid * 16, **model_kwargs)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model loaded successfully')
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs)
    scaler = dataloader['scaler']
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    met_df, yhat = util.calc_test_metrics(model, device, dataloader['test_loader'], scaler, realy)
    df2 = util.make_pred_df(realy, yhat, scaler)
    met_df.to_csv('last_test_metrics.csv')
    df2.to_csv('./wave.csv', index=False)


    if args.plotheatmap == "True":
        plot_learned_adj_matrix(model)
    return met_df, df2

def plot_learned_adj_matrix(model):
    adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    adp = adp.cpu().detach().numpy()
    adp = adp / np.max(adp)
    df = pd.DataFrame(adp)
    sns.heatmap(df, cmap="RdYlBu")
    plt.savefig("heatmap.png")


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    parser.add_argument('--checkpoint', type=str, help='')
    parser.add_argument('--plotheatmap', action='store_true')

    args = parser.parse_args()
    main(args)
