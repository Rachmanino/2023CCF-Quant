import torch.nn as nn
import torch

from config import Config


class Batch_Loss(nn.Module):
    def __init__(self, config:Config):
        super(Batch_Loss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.EPS = 1e-6


        self.loss_mode = config.loss_mode


    ####使用MSE来计算之间的相关性,目标是使得相关性的绝对值最大。
    def forward(self, preds, targets):


        target_next_time = targets[:,0]
        target_next_period = targets[:, 1]


        preds_next_time = preds[:,0]
        preds_next_period = preds[:, 1]
        loss = self.loss_fn(preds, targets)

        if self.loss_mode==1:
            loss += self.loss_fn(preds_next_time - preds_next_period, target_next_time - target_next_period)
        assert not torch.isnan(loss).any(), "Loss is NaN!"
        assert not torch.isinf(loss).any(), "Loss is INF!"
        if abs(loss)>1e10:{
            print(loss)}
        return loss


