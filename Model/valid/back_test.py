import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import pickle
import sys

########## set default arguments
sys.path.append('..')
from dataset.stock_dataset import Stock_Dataset
from train.batch_loss import Batch_Loss

model_save_dir = "./loadmodule/model_end.pt"
config_save_dir = "./loadmodule/config.obj"
THRESHOLDS = 0.01

#####load config
with open(config_save_dir, 'rb') as f:
    config = pickle.load(f)
# print('config load successful')


#####load model
model = torch.load(model_save_dir, map_location=config.device)

data_dir = f"../{config.data_dir}"
train_dataset = Stock_Dataset(data_dir=data_dir,day_windows=config.day_windows+1,
                              flag="train", device=config.device)
valid_dataset = Stock_Dataset(data_dir=data_dir,day_windows=config.day_windows+1,
                              flag="valid", device=config.device)
# test_dataset = Stock_Dataset(data_dir=data_dir,day_windows=config.day_windows,
#                               flag="test", device=config.device)

# DataSets = [train_dataset, valid_dataset, test_dataset]
DataSets = [train_dataset, valid_dataset]

batch_loss = Batch_Loss(config)# MODE = ["train", "valid", "test"]

MODE = ["train", "valid"]


with torch.no_grad():
    model.eval()
    for i in range(len(DataSets)):
        dataset = DataSets[i]
        mode = MODE[i]
        preds_list = []
        loss = 0
        is_hold = False
        last_prices = None
        pnl_list = []
        loss_list = []

        for data, target in dataset:
            pnl = 0
            print(type(data[:-1].unsqueeze(0)))
            preds = model(data[:-1].unsqueeze(0))[0]


            loss = batch_loss(preds, target)

            loss_list.append(loss.cpu().detach().numpy())


            preds_list.append(preds.cpu().detach().numpy())

            preds_next_time = preds[0]
            preds_next_period = preds[1]

            if(preds_next_time - preds_next_period> THRESHOLDS):
                if not is_hold:
                    if last_prices:
                        pnl = (last_prices -  data[-1, 4])/last_prices
                        pnl = pnl.cpu().detach().numpy()
                    else:
                        last_prices = data[-1, 4]
            if (preds_next_time - preds_next_period < -THRESHOLDS):
                if is_hold:
                    if last_prices:
                        pnl = -(last_prices - data[-1, 5])/last_prices
                        pnl = pnl.cpu().detach().numpy()
                    else:
                        last_prices = data[-1, 5]

            pnl_list.append(pnl)
        print("******************************************")
        print(f"{MODE[i]} dataset loss: { np.stack(loss_list).mean()}")

        plt.plot([i for i in range(len(pnl_list))], pnl_list, label="pnl")
        plt.legend()
        plt.savefig(f"{MODE[i]} pnl.png")


        plt.clf()