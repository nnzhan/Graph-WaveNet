import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


def nconv(x, A):
    """Multiply along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, apt_size=10):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.addaptadj = addaptadj

        # Each of these will
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if do_graph_conv and addaptadj:
            if supports is None: self.supports = []
            if aptinit is None:
                nodevec1 = torch.randn(num_nodes, apt_size)
                nodevec2 = torch.randn(apt_size, num_nodes)
            else:
                m, p, n = torch.svd(aptinit)
                nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
                nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
            self.supports_len += 1
            
            self.register_parameter('nodevec1', nn.Parameter(nodevec1.to(device), requires_grad=True))
            self.register_parameter('nodevec2', nn.Parameter(nodevec2.to(device), requires_grad=True))

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.do_graph_conv:
                    self.graph_convs.append(
                        GraphConvNet(dilation_channels, residual_channels, dropout,
                                     support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        if self.do_graph_conv and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adjacency_matrices = self.supports + [adp]
        else:
            adjacency_matrices = self.supports

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
            s = x
            s = self.skip_convs[i](s)
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip = 0
            skip = s + skip

            if self.do_graph_conv and self.supports is not None:
                support = adjacency_matrices if self.addaptadj else self.supports
                x = self.graph_convs[i](x, support)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):] # TODO(SS): Mean/Max Pool?
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





