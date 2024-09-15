import torch
import numpy as np
import argparse
import glob
import os
import time
import util
import matplotlib.pyplot as plt
import pandas as pd
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
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
args = parser.parse_args()




def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    patience = 1000
    epochs_since_best_rmse = 0
    lowest_rmse_yet = 100
    best_model_save_path = os.path.join(args.save, 'best_model.pth')
    os.makedirs(args.save, exist_ok=True)

    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.splits, args.prediction_multi_or_single, args.single_prediction_time_step)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit) #.to(device)


    print("start training...",flush=True)
    his_loss =[]
    total_val_loss =[]
    total_mean_val_loss =[]
    total_val_rmse =[]
    total_mean_val_rmse =[]
    val_time = []
    train_time = []
    total_train_loss = []
    total_train_rmse = []
    # val_realy = torch.Tensor(dataloader['y_val']).to(device)
    # val_realy = val_realy.transpose(1,3)[:,0,:,:]
    if (not args.no_train):
        for i in range(args.from_epochs + 1,args.epochs+1):
            print(f'Epoch number: {i}')
            #if i % 10 == 0:
                #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #for g in engine.optimizer.param_groups:
                    #g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y, _, _) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)

                if args.prediction_multi_or_single=="single":
                    # select specified temporal horizon
                    trainy = trainy[:,:,:, args.single_prediction_time_step-1].unsqueeze(-1)
                    metrics = engine.train(trainx, trainy[:, 0, :, :])

                else:
                    # If using multiple horizons, trainy still has 4 dimensions
                    metrics = engine.train(trainx, trainy[:, 0, :, :])


                train_loss.append(metrics[0])
                total_train_rmse.append(metrics[2])
                total_train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []

        
            s1 = time.time()
            for iter, (x, y, _, _) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)

                if args.prediction_multi_or_single=="single":
                    # select specified temporal horizon 
                    testy = testy[:,:,:,args.single_prediction_time_step - 1].unsqueeze(-1)
                    metrics = engine.eval(testx, testy[:,0,:,:])

                else:
                    metrics = engine.eval(testx, testy[:,0,:,:]) 
                    
                valid_loss.append(metrics[0])
                total_val_loss.append(metrics[0])
                total_val_rmse.append(metrics[2])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])

            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            
            total_mean_val_loss.append(mvalid_loss)
            total_mean_val_rmse.append(mvalid_rmse)
            if mvalid_rmse < lowest_rmse_yet:
                torch.save(engine.model.state_dict(), best_model_save_path)
                lowest_rmse_yet = mvalid_rmse
                epochs_since_best_rmse = 0
            else:
                epochs_since_best_rmse += 1

            if epochs_since_best_rmse >= patience:
                print('patience ', patience)
                print('epochs_since_best_rmse ', epochs_since_best_rmse)
                print(f'current_valid_loss: {mvalid_rmse} epochs: {i}')
                break

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

        print()
        print()
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        print() 
        print()

        train_loss_file = open("./garage/train_loss.txt", "w")
        val_loss_file = open("./garage/val_loss.txt", "w")
        mean_val_loss_file = open("./garage/mean_val_loss.txt", "w")
        train_rmse_file = open("./garage/train_rmse.txt", "w")
        val_rmse_file = open("./garage/val_rmse.txt", "w")
        mean_val_rmse_file = open("./garage/mean_val_rmse.txt", "w")

        for element in total_train_loss:
            train_loss_file.write(str(element) + "\n")
        
        for element in total_mean_val_loss:
            mean_val_loss_file.write(str(element) + "\n")

        for element in total_val_loss:
            val_loss_file.write(str(element) + "\n")

        for element in total_train_rmse:
            train_rmse_file.write(str(element) + "\n")
        
        for element in total_mean_val_rmse:
            mean_val_rmse_file.write(str(element) + "\n")

        for element in total_val_rmse:
            val_rmse_file.write(str(element) + "\n")
            

        train_loss_file.close()
        val_loss_file.close()
        mean_val_loss_file.close()
        train_rmse_file.close()
        val_rmse_file.close()
        mean_val_rmse_file.close()

    #testing
    if (not args.no_train):
        bestid = np.argmin(total_mean_val_loss)
        # engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(total_mean_val_loss[bestid],2))+".pth"))
    else:
        list_of_files = glob.glob('./garage/*best?*')
        latest_file = max(list_of_files, key=os.path.getctime)
        engine.model.load_state_dict(torch.load(latest_file))

    if (len(engine.model.supports) != 0):
        adapted_adj_matrix = engine.model.supports[0].numpy() if (args.device == 'cpu') else engine.model.supports[0].cpu().numpy()
        np.save("./garage/adj_adpt", adapted_adj_matrix)

    outputs = []

    if isinstance(dataloader['y_test'], torch.Tensor):
        realy = dataloader['y_test'].clone().detach().to(dtype=torch.float32, device=device)
    else:
        realy = torch.tensor(dataloader['y_test'], dtype=torch.float32).to(device)

    realy = realy.transpose(1,3)[:,0,:,:]

    dates = dataloader['dates_test']
    stations = dataloader['stations_test']

    for iter, (x, y, _, _) in enumerate(dataloader['test_loader'].get_iterator()):

        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)

        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())    


    # Print the shape of x and y to ensure they're being loaded
    print()
    print(f"Prediction type: {args.prediction_multi_or_single} , Shape of trainx: {trainx.shape}, Shape of trainy: {trainy.shape}")
    print()    
    
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")

    
    if (not args.no_train):
        print("The valid loss on best model is", str(round(total_mean_val_loss[bestid],4)))

    
    amae = []
    amape = []
    armse = []

    
    if args.prediction_multi_or_single=='single':

        print()
        print(f'Yhat: {yhat.shape} , Yrealy: {realy.shape}')
        print()
        pred = scaler.inverse_transform(yhat)
        print()
        print(f"Shape of pred: {pred.shape}, Shape of realy: {realy.shape}, Shape of realy sliced to relevant temporal horizon only: {realy[:,:,args.single_prediction_time_step - 1].shape}")
        print()
        metrics = util.metric(pred, realy[:,:,args.single_prediction_time_step - 1]) # Consider only the relevant temporal horizon 
        print("Keep only the desired temporal horizon in realy to match the shape of pred")
        print(f"Shape of pred: {pred.shape}, Shape of realy: {realy.shape}")
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.single_prediction_time_step, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

        log = 'For horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.single_prediction_time_step, np.mean(amae), np.mean(amape), np.mean(armse)))
    
    else:

        
        if args.seq_length==1:
            end_range = 1
        else:
            end_range = args.seq_length
            
        for i in range(args.from_seq_length,end_range):

            pred = scaler.inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
                
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over {:.4f} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.seq_length, np.mean(amae),np.mean(amape),np.mean(armse)))     
    
    
    path_name = args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth"
    
    if (not args.no_train):
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(total_mean_val_loss[bestid],2))+".pth")
    else:
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best.pth")

    return path_name

def save_metrics(train_loss, mean_val_loss, val_loss, train_rmse, mean_val_rmse, val_rmse):
    os.makedirs("./garage", exist_ok=True)
    save_to_file("./garage/train_loss.txt", train_loss)
    save_to_file("./garage/mean_val_loss.txt", mean_val_loss)
    save_to_file("./garage/val_loss.txt", val_loss)
    save_to_file("./garage/train_rmse.txt", train_rmse)
    save_to_file("./garage/mean_val_rmse.txt", mean_val_rmse)
    save_to_file("./garage/val_rmse.txt", val_rmse)

def save_to_file(filename, data):
    with open(filename, "w") as f:
        for element in data:
            f.write(str(element) + "\n")

if __name__ == "__main__":

    t1 = time.time()
    path_name = main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
    print("Checkpoint save file: {}".format(path_name))
