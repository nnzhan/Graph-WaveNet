import torch.optim as optim
from model import *
import util
class Trainer():
    def __init__(self, model, scaler, lrate, wdecay, clip=5):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3) # output = [batch_size,12,num_nodes,1]
        predict = self.scaler.inverse_transform(output)

        real = torch.unsqueeze(real_val, dim=1)
        mae, mape, rmse = util.cheaper_metric(predict, real, null_val=0.0)
        mae.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return mae.item(),mape.item(),rmse.item()

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3) #  [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        mae, mape, rmse = [x.item() for x in util.cheaper_metric(predict, real, null_val=0.0)]
        return mae, mape, rmse
