import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 softmax_temp=1., addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 apt_size=10):
        super().__init__()
        self.dropout = dropout
        self.softmax_temp = softmax_temp
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.addaptadj = addaptadj

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevec1, nodevec2 = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevec1, nodevec2 = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.register_parameter('nodevec1', Parameter(nodevec1.to(device), requires_grad=True))
            self.register_parameter('nodevec2', Parameter(nodevec2.to(device), requires_grad=True))

        depth = blocks * layers

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in range(depth)])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in range(depth)])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in range(depth)])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                                              for _ in range(depth)])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1,kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        model = cls(device, args.num_nodes, dropout=args.dropout, supports=supports,
                    do_graph_conv=args.do_graph_conv, addaptadj=args.addaptadj, aptinit=aptinit,
                    in_dim=args.in_dim, apt_size=args.apt_size, out_dim=args.seq_length,
                    softmax_temp=args.softmax_temp,
                    residual_channels=args.nhid, dilation_channels=args.nhid,
                    skip_channels=args.nhid * 8, end_channels=args.nhid * 16, **kwargs)
        return model

    def forward(self, input):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        if input.size(3) < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - input.size(3), 0, 0, 0))
        x = self.start_conv(input)
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)) / self.softmax_temp, dim=1)
            adjacency_matrices = self.fixed_supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*input*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i](x)  # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            if self.do_graph_conv:
                x = self.graph_convs[i](x, adjacency_matrices)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):] # TODO(SS): Mean/Max Pool?
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





