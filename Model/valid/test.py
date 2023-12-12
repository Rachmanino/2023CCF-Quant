import math
from tracemalloc import start
import torch
import argparse
import numpy as np
import pickle
import sys
import time
import os
from tqdm import tqdm
########## set default arguments
sys.path.append('..')
from dataset.stock_dataset import Stock_Dataset

model_save_dir = "./loadmodule/model_end.pt"
config_save_dir = "./loadmodule/config.obj"


#####load config
with open(config_save_dir, 'rb') as f:
    config = pickle.load(f)

#####load model
model = torch.load(model_save_dir, map_location=config.device)

#####construct dataset
DataSets = []
csv_names = [csv_name[:-4] for csv_name in os.listdir('../../data/20231128')]
for csv_name in csv_names:
    DataSets.append(Stock_Dataset(data_dir='../../data/20231128', day_windows=config.day_windows+1,
                                  flag=csv_name, device=config.device))

model.eval()
PNL = 0
TRADES = 0
WINS = 0
PNLS = np.array([])

start = time.time()
for i in tqdm(range(len(DataSets))):
    dataset = DataSets[i]
    preds_list = []

    MAXTIME = 10000
    last_prices = 100
    last = None
    cnt = 0

    # print(csv_names[i])

    for j in range(len(dataset)):
        data, target = dataset[j]
        THRESHOLDS = 0.01
        pnl = 0

        with torch.no_grad():
            preds = model(data[:-1].unsqueeze(0))[0]

        preds_list.append(preds.cpu().detach().numpy())

        preds_next_time = preds[0]
        preds_next_period = preds[1]
        # print("p-p", (preds_next_time-preds_next_period).tolist())

        ###################### OUR STRATEGY #######################

        cnt += 1
        # print(data, cnt)
        if last != 'sell' and \
                ((preds_next_time - preds_next_period > THRESHOLDS 
                and data[-2,0] * (1.0 + preds_next_time/100) > last_prices)
                or cnt >= MAXTIME):
            if last == 'buy':
                TRADES += 1
            cnt = 0
            if last == None:
                last_prices = data[-1, 4]
            else:
                pnl = (data[-1, 4] - last_prices) / last_prices
                last_prices = data[-1, 4]
            last = 'sell'
            print(' sell, pnl=', pnl*1e4)
        elif last != 'buy' and \
                ((preds_next_period - preds_next_time > THRESHOLDS 
                and data[-2,0] * (1.0 + preds_next_time/100) < last_prices)
                or cnt >= MAXTIME):
            if last == 'sell':
                TRADES += 1
            cnt = 0
            if last == None:   # buy
                last_prices = data[-1, 4]
            else:
                pnl = (last_prices - data[-1, 4]) / last_prices
                last_prices = data[-1, 4]
            last = 'buy'
            # print(' buy, pnl=', pnl*1e4)
        ###################### END STRATEGY #######################
        if j == len(dataset)-1:   # 强制平仓
            if last == 'buy':
                TRADES += 1
                # 强制卖出
                pnl = (data[-1, 4] - last_prices) / last_prices
                # print('force: sell, pnl=', pnl*1e4)
            elif last == 'sell':
                TRADES += 1
                # 强制买入
                pnl = (last_prices - data[-1, 4]) / last_prices
                # print('force: buy, pnl=', pnl*1e4)

        if pnl != 0:
            PNL += pnl    
            PNLS = np.append(PNLS, pnl.cpu().detach().numpy())
        # print(PNLS)
        if pnl > 0:
            WINS += 1
            
    if (i+1) % 10 == 0:
        print(f'{i} / {len(csv_names)}, pnl={PNL * 1e4/TRADES:.4f}, win={WINS / TRADES:.4f}, trade={TRADES}')

WIN_RATE = WINS / TRADES
PNL = PNL * 1e4 / TRADES
SCORE = PNL * WIN_RATE * math.log(TRADES)

# log
end = time.time()
print('****************************************************************************')
print(f"For THERSHOLD = {THRESHOLDS}, after {end-start}s")
print('  pnl:', PNL)
print('  win rate:', WIN_RATE)
print('  trades:', TRADES)
print('score:', SCORE)
print('pnl_avg:', PNLS.mean())
print('pnl_std:', PNLS.std())
print('sharpe_ratio:', PNLS.mean() / PNLS.std())