import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import argparse
import numpy as np
import pickle
import os

########## set default arguments
from tqdm import tqdm
sys.path.append('..')
from dataset.stock_dataset import Stock_Dataset
from train.batch_loss import Batch_Loss
from utils import get_loaders


model_save_dir = "./loadmodule/model.pt"
config_save_dir = "./loadmodule/config.obj"
THRESHOLDS = 0.01;


#####load config
with open(config_save_dir, 'rb') as f:
    config =pickle.load(f)
config.batch_size = 1
config.device = 'cuda:0'
#####load model
model = torch.load(model_save_dir, map_location=config.device)


config.data_dir = "../dataset/data"

train_dataloader_list,  valid_dataloader_list= get_loaders(config=config)

batch_loss = Batch_Loss(config)# MODE = ["train", "valid", "test"]

MODE = ["train", "valid"]

for data_loader in valid_dataloader_list:
    dataset = data_loader.dataset
    with torch.no_grad():
        model.eval()
        print("******************************************************************"+dataset.name)


        for idx, (data, embedding_intput , targets) in enumerate(tqdm(data_loader)):
            pnl = None
            print(data.shape, embedding_intput)
            preds = model(data, embedding_intput)


            # loss = batch_loss(preds, target)
            #
            # loss_list.append(loss.cpu().detach().numpy())


            # preds_list.append(preds.cpu().detach().numpy())

            preds_next_time = preds[:,0]
            preds_next_period = preds[:,1]
            print("target" + str(torch.sum(targets[:,0] - targets[:,1]>0).item()))

            print("preds" +str(torch.sum(preds_next_period - preds_next_time > 0).item()))

            #     if(preds_next_time - preds_next_period> THRESHOLDS):
            #         if not is_hold:
            #             if last_prices:
            #                 pnl = (last_prices - data[-1, 4])/last_prices
            #                 pnl = pnl.cpu().detach().numpy()
            #             else:
            #                 last_prices = data[-1, 4]
            #     if (preds_next_time - preds_next_period < -THRESHOLDS):
            #         if is_hold:
            #             if last_prices:
            #                 pnl = -(last_prices - data[-1, 4])/last_prices
            #                 pnl = pnl.cpu().detach().numpy()
            #             else:
            #                 last_prices = data[-1, 4]
            #     if pnl:
            #         if pnl > 0: wintrades += 1
            #         pnl_list.append(pnl)
            #     totaltrades += 1
            # print("******************************************")
            # print(f"{MODE[i]} dataset loss: { np.stack(loss_list).mean()}")
            #
            # plt.plot([i for i in range(len(pnl_list))], pnl_list, label="pnl")
            # plt.legend()
            # plt.savefig(f"{MODE[i]} pnl.png")
            #
            #
            # plt.clf()












