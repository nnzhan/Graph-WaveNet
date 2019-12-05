import torch.optim as optim
from model import *
import util

class Trainer():
    def __init__(self, model: GWNet, scaler, lrate, wdecay, clip=5, lr_decay_rate=.97, fp16='', end_conv_lr=None):
        self.model = model
        if end_conv_lr:
            end_conv2, other_params = model.conv_group
            groups = [{'params': end_conv2, 'lr': end_conv_lr}]
            if lrate > 0:
                groups.append({'params': other_params})
            self.model.freeze_group_b()
            self.optimizer = optim.Adam(groups, lr=lrate, weight_decay=wdecay)
        else:
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
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16)

    @classmethod
    def from_args(cls, model, scaler, args):
        end_conv_lr = getattr(args, 'end_conv_lr', None)
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate, fp16=args.fp16, end_conv_lr=end_conv_lr)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input).transpose(1,3)  # now, output = [batch_size,1,num_nodes, seq_length]
        predict = self.scaler.inverse_transform(output)
        assert predict.shape[1] == 1
        mae, mape, rmse = util.calc_metrics(predict.squeeze(1), real_val, null_val=0.0)

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
