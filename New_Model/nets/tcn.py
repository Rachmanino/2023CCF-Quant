import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from config import Config
from nets.mha import MHA


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Tcn_Module(nn.Module):
    def __init__(self, args:Config):
        super(Tcn_Module, self).__init__()

        num_channels = [args.tcn_hidden_dim] * args.tcn_levels

        tcn_drop = 0.2
        self.tcn = TemporalConvNet(num_inputs= args.indicator_num, num_channels=num_channels, kernel_size=args.tcn_kernel_size, dropout=args.drop)
        self.drop = nn.Dropout(args.drop)

        self.wind1_embedding_layer = nn.Embedding(num_embeddings=args.wind1_embeddings, embedding_dim=args.tcn_hidden_dim)
        self.wind2_embedding_layer = nn.Embedding(num_embeddings=args.wind2_embeddings, embedding_dim=args.tcn_hidden_dim)
        self.company_embedding_layer = nn.Embedding(num_embeddings=args.company_embeddings, embedding_dim=args.tcn_hidden_dim)

        self.mha =nn.ModuleList([MHA(args) for _ in range(args.mha_layers)])




        self.mlp = nn.Linear(in_features= 4 * args.tcn_hidden_dim, out_features=2)



    def forward(self, stock_info, embedding_intput):
        # stock_info:  [batch,catch_len, input_dim]

        batch = stock_info.shape[0]
        catch_len = stock_info.shape[1]
        input_dim = stock_info.shape[2]
        # stock_info:  [catch_len, stock_num, input_dim]
        # data_trans:  [batch, input_dim,catch_len ]
        data_trans = stock_info.permute(0,2,1).contiguous()

        data_trans_no_index = data_trans[:,:-1,:]

        # data_trans:  [batch, 1,catch_len ]
        tcn_out = self.tcn(data_trans_no_index)[:, :, -1]
        ##embedded_price  shape [batch, tcn_embeddings]

        ##embedded_wind1  shape embedded_wind1
        embedded_wind1 = self.wind1_embedding_layer(embedding_intput[0]).unsqueeze(1)
        embedded_wind2 = self.wind2_embedding_layer(embedding_intput[1]).unsqueeze(1)
        embedded_company = self.company_embedding_layer(embedding_intput[2]).unsqueeze(1)

        embedding_input = torch.cat((embedded_wind1, embedded_wind2, embedded_company),dim=1)

        ##mhd_out  shape [batch,3, tcn_embeddings]
        mhd_out = embedding_input
        for layer in self.mha:
            mhd_out = layer(mhd_out)


        tcn_out = tcn_out.unsqueeze(1)
        combin = torch.cat((tcn_out,mhd_out),dim=1)
        combin = combin.view((batch, -1));

        out = self.mlp( self.drop(combin))



        return out