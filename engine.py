import torch.optim as optim
from model import *
import util

class Trainer():
    def __init__(self, model: GWNet, scaler, lrate, wdecay, clip=3, lr_decay_rate=.97):
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    @classmethod
    def from_args(cls, model, scaler, args):
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3)  # now, output = [batch_size,1,num_nodes, seq_length]
        predict = self.scaler.inverse_transform(output)
        assert predict.shape[1] == 1
        mae, mape, rmse = util.calc_metrics(predict.squeeze(1), real_val, null_val=0.0)
        mae.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return mae.item(),mape.item(),rmse.item()

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3) #  [batch_size,seq_length,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=70.)
        mae, mape, rmse = [x.item() for x in util.calc_metrics(predict, real, null_val=0.0)]
        return mae, mape, rmse
