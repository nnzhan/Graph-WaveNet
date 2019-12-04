import torch.optim as optim
from model import *
import util

class Trainer():
    def __init__(self, model, scaler, lrate, wdecay, clip=5, lr_decay_rate=.97, fp16=''):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip
        self.fp16 = fp16
        l1 = lambda epoch: lr_decay_rate ** epoch
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l1)
        if self.fp16:
            try:
                from apex import amp  # Apex is only required if we use fp16 training
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            amp.register_half_function(torch, 'einsum')
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                            opt_level=self.fp16)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3)  # now, output = [batch_size,1,num_nodes,seq_length]

        #torch.clamp(output, 0, 70)

        predict = self.scaler.inverse_transform(output)
        real = torch.unsqueeze(real_val, dim=1)
        mae, mape, rmse = util.calc_metrics(predict, real, null_val=0.0)

        if self.fp16:
            from apex import amp
            with amp.scale_loss(mae, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.clip)
        else:
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
