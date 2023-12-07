import math
from tracemalloc import start
import torch
import argparse
import numpy as np
import pickle
import sys
import time
import os
########## set default arguments
sys.path.append('..')
from dataset.stock_dataset import Stock_Dataset
from train.batch_loss import Batch_Loss

model_save_dir = "./loadmodule/model.pt"
config_save_dir = "./loadmodule/config.obj"

#####load config
with open(config_save_dir, 'rb') as f:
    config = pickle.load(f)

#####load model
model = torch.load(model_save_dir, map_location=config.device)

#####construct dataset
DataSets = []
csv_names = [csv_name[:-4] for csv_name in os.listdir('../../data/20231120')]
for csv_name in csv_names:
    if csv_name == '000009.SZ':
        DataSets.append(Stock_Dataset(data_dir='../../data/20231120',day_windows=config.day_windows+1,
                              flag=csv_name, device=config.device, name=csv_name))
print("DataSets:", len(DataSets))
print(csv_names[0])
# TODO:construct dataset

with torch.no_grad():
    model.eval()
    PNL = 0
    TRADES = 0
    WINS = 0
    start = time.time()
    for i in range(1):
        dataset = DataSets[i]
        preds_list = []
        is_hold = False
        last_prices = None
        pnl_list = []
        loss_list = []

        for data, embedding_intput, target in dataset:
            pnl = 0
            preds = model(data[:-1].unsqueeze(0), embedding_intput)[0]

            preds_list.append(preds.cpu().detach().numpy())

            preds_next_time = preds[0]
            preds_next_period = preds[1]
            print("preds_next_time-preds_next_period", preds_next_time-preds_next_period)

            ###################### OUR STRATEGY #######################
            THRESHOLDS = 0.003
            # print(preds_next_time, preds_next_period, data[-1])
            if (preds_next_time - preds_next_period> THRESHOLDS):
                if not is_hold:     # buy
                    TRADES += 1
                    if last_prices:
                        pnl = (last_prices -  data[-1, 4])/last_prices
                    else:
                        last_prices = data[-1, 4]
            if (preds_next_time - preds_next_period < -THRESHOLDS):
                if is_hold:
                    TRADES += 1
                    if last_prices:
                        pnl = -(last_prices - data[-1, 4])/last_prices
                    else:
                        last_prices = data[-1, 4]
            ###################### END STRATEGY #######################
                        
            if pnl > 0:
                WINS += 1
            PNL += pnl
    
    WIN_RATE = WINS / TRADES
    PNL = PNL * 1e4 / TRADES
    SCORE = PNL * WIN_RATE * math.log(TRADES) 

    # log
    end = time.time()
    print(f"For THERSHOLD = {THRESHOLDS}, after {end-start}s")
    print('  pnl:', PNL.tolist())
    print('  win rate:', WIN_RATE)
    print('  trades:', TRADES)
    print('score:', SCORE.tolist())
