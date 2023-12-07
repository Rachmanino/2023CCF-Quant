import torch.nn as nn
import torch

from config import Config


class Batch_Loss(nn.Module):
    def __init__(self, config:Config):
        super(Batch_Loss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.EPS = 1e-6


    ####使用MSE来计算之间的相关性,目标是使得相关性的绝对值最大。
    def forward(self, preds, targets):
        loss = self.loss_fn(preds, targets)

        if (abs(loss)>1e10):{
            print(loss)}
        return loss


