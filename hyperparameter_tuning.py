import optuna
import subprocess
import argparse

def objective(trial, args):
    # Hyperparameters to be optimized
    lrate = trial.suggest_loguniform("lrate", args.lrate_min, args.lrate_max)
    wdecay = trial.suggest_loguniform("wdecay", args.wdecay_min, args.wdecay_max)
    dropout = trial.suggest_uniform("dropout", args.dropout_min, args.dropout_max)
    batch_sizes = trial.suggest_categorical("batch_sizes", args.batch_sizes)
    
    # Construct command
    cmd = (f'python Graph-WaveNet/train.py '
           f'--device={args.device} '
           f'--data={args.data} '
           f'--adjdata={args.adjdata} '
           f'--adjtype={args.adjtype} '
           f'--aptonly={args.aptonly} '
           f'--addaptadj={args.addaptadj} '
           f'--gcn_bool={args.gcn_bool} '
           f'--randomadj={args.randomadj} '
           f'--seq_length={args.seq_length} '
           f'--from_seq_length={args.from_seq_length} '
           f'--nhid={args.nhid} '
           f'--in_dim={args.in_dim} '
           f'--num_nodes={args.num_nodes} '
           f'--batch_size={batch_sizes} '
           f'--learning_rate={lrate} '
           f'--dropout={dropout} '
           f'--weight_decay={wdecay} '
           f'--epochs={args.epochs} '
           f'--from_epochs={args.from_epochs} '
           f'--print_every={args.print_every} '
           f'--save={args.save} '
           f'--expid={args.expid} '
           f'--no_train={args.no_train} '
           f'--prediction_multi_or_single={args.prediction_multi_or_single} '
           f'--single_prediction_time_step={args.single_prediction_time_step} '
           f'--splits={args.splits}')

    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Extract the relevant line from stdout and parse the validation loss
    output_lines = result.stdout.decode().split("\\n")
    valid_line = [line for line in output_lines if "The valid loss on best model is" in line][0]
    val_loss = float(valid_line.split()[-1])
    #val_loss = float(valid_loss_best)
    
    return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Define all the command-line arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--data',type=str,default='data/MAX-TEMP',help='data path')
    parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')  # Este parametro dice si inicializamos el tercer termino con info de las estaciones
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj') # Este parametro dice si incluimos unicamente el tercer termino
    parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj') # Este parametro dice si el tercer termino aprende o no 
    parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj') # Este parametro dice si inicializamos el tercer termino randomicamente o no
    parser.add_argument('--seq_length',type=int,default=12,help='output channels for the final convolution layer conv2d( ), determines the number of time steps that will be predicted forward. If you want multiple predictions (e.g. t+1, t+2, t+3) then seq_length=3, if you want only a single prediction (e.g. only t+3) then seq_length=1') # son los y
    parser.add_argument('--from_seq_length',type=int,default=0,help='') # para acotar el rango de horizontes temporales a predecir
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--epochs',type=int,default=20,help='')
    parser.add_argument('--from_epochs',type=int,default=0,help='')
    parser.add_argument('--print_every',type=int,default=50,help='')
    #parser.add_argument('--seed',type=int,default=99,help='random seed')
    parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')
    parser.add_argument('--no_train', action='store_true',help='use the last saved model')
    parser.add_argument('--prediction_multi_or_single',type=str,default='multi', help='indicate the problem as either multi or single, to determine whether predictions are done for multiple time steps sequentially (has generality, less precision) or for a unique time step (more precision, no generality)')
    parser.add_argument('--single_prediction_time_step',type=int,default=None, help='indicate which single temporal horizon to be used')
    parser.add_argument('--splits',type=int,default=10, help='splits for timesteps')

    parser.add_argument('--lrate_min', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--lrate_max', type=float, default=1e-1, help='Maximum learning rate')
    parser.add_argument('--wdecay_min', type=float, default=1e-5, help='Minimum weight decay')
    parser.add_argument('--wdecay_max', type=float, default=1e-1, help='Maximum weight decay')
    parser.add_argument('--dropout_min', type=float, default=0, help='Minimum dropout')
    parser.add_argument('--dropout_max', type=float, default=1, help='Maximum dropout')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32, 64, 128], help='List of batch sizes to choose from')

   
    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=10)
