import torch
import torch.nn as nn
import torch.nn.functional as F


class Lstm_Module(nn.Module):
    def __init__(self, args:Config):
        super(Lstm_Module, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=args.lstm_input_dim, hidden_size=args.lstm_hidden_size)
        self.h_init = torch.nn.Parameter(torch.zeros(1, args.stock_num, args.lstm_hidden_size), requires_grad=True)
        self.c_init = torch.nn.Parameter(torch.zeros(1, args.stock_num, args.lstm_hidden_size), requires_grad=True)

        self.W1 = nn.Linear(args.lstm_hidden_size, args.embedding_dim, bias=False)
        self.W2 = nn.Linear(args.lstm_hidden_size, args.embedding_dim, bias=False)
        self.W3 = nn.Linear(args.lstm_hidden_size, 1, bias=False)

        self.embedding_dim = args.embedding_dim
        self.stock_num = args.stock_num
        self.cache_day_len = args.cache_day
        self.lstm_input_dim = args.lstm_input_dim


    def forward(self, stock_info):
        #stock_info:  [batch,catch_len, stock_num, input_dim]

        batch = stock_info.shape[0]
        catch = stock_info.shape[1]
        stock_num = stock_info.shape[2]
        # stock_info:  [catch_len, stock_num, input_dim]

        data_trans = stock_info.transpose(dim0=1, dim1=2).contiguous().view(stock_num * batch, catch, -1).transpose(dim0=0, dim1=1)

        # output [catch_len, stock_num* batch, embedding_dim],   hn  [1, stock_num * batch, embedding_dim]
        output, (hn, cn) = self.lstm_layer(data_trans)

        ##output_trans [batch,stock_num, catch_len, embedding_dim]    hn_trans  [batch,stock_num, 1, embedding_dim]

        output_trans = torch.transpose(output, 0, 1).view(batch, stock_num, catch, 128)
        hn_trans = torch.transpose(hn, 0, 1).view(batch, stock_num, 1, 128)

        # temp [batch,stock_num, catch_len, embedding_dim]
        temp = self.W1(output_trans) + self.W2(hn_trans).tanh()

        # apltha  [batch,stock_num, catch_len ]
        alpha = self.W3(temp).squeeze(dim=3)

        # soft_alpha = [batch,stock_num,catch_len,1]
        soft_alpha = F.softmax(alpha, dim=2)
        soft_alpha = soft_alpha.unsqueeze(3)

        # result = [batch,stock_num, embedding_dim]
        result = torch.mul(output_trans, soft_alpha).sum(dim=2)
        return result