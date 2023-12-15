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
# set default arguments
sys.path.append('..')
from dataset.stock_dataset import Stock_Dataset

model_save_dir = "./new_model/LossMode0/model.pt"
config_save_dir = "./new_model/LossMode0/config.obj"


# load config
with open(config_save_dir, 'rb') as f:
    config = pickle.load(f)

# load model
model = torch.load(model_save_dir, map_location=config.device)

# construct dataset
DataSets = []
csv_names = [csv_name[:-4] for csv_name in os.listdir('../../data/20231121')]
for csv_name in csv_names:
    print(csv_name)
    DataSets.append(Stock_Dataset(data_dir='../../data/20231124', name = csv_name,day_windows=config.day_windows+1,
                                  flag=csv_name, device=config.device))

model.eval()
PNL = 0
TRADES = 0
WINS = 0
PNLS = np.array([])
revenue = 0

start = time.time()
for i in tqdm(range(len(DataSets))):
    dataset = DataSets[i]
    preds_list = []

    MAXTIME = 10000
    THRESHOLDS = 0.01
    last_prices = 100
    last_data_close = 0
    last = None
    cnt = 0

    # print(csv_names[i])

    for j in range(len(dataset)):
        data, embedding_intput, target, data_close = dataset[j]
        
        pnl = 0

        with torch.no_grad():
            preds = model(data[:-1].unsqueeze(0), embedding_intput)[0]

        preds_list.append(preds.cpu().detach().numpy())

        preds_next_time = preds[0]
        preds_next_period = preds[1]

        ###################### OUR STRATEGY #######################
        if j != len(dataset)-1:
            # print(data, cnt)
            if last != 'sell' and \
                    ((data[-2, 0] * (1.0 + preds_next_time/100) > last_prices)
                    or cnt <= -MAXTIME):
                if last == 'buy':
                    TRADES += 1
                    pnl = (data[-1, 4] * data_close - last_prices * last_data_close) / (last_prices * last_data_close)
                    revenue += (data[-1, 4] * data_close - last_prices * last_data_close)
                last = 'sell'
                last_prices = data[-1, 4] 
                last_data_close = data_close
                # print('p-p', (preds_next_time-preds_next_period).tolist())
                # print(' sell, pnl=', pnl*1e4, j)
            elif last != 'buy' and \
                    ((data[-2, 0]  * (1.0 + preds_next_time/100) < last_prices)
                    or cnt >= MAXTIME):
                if last == 'sell':
                    TRADES += 1
                    pnl = (last_prices * last_data_close - data[-1, 4] * data_close) / (last_prices * last_data_close)
                    revenue += (last_prices * last_data_close - data[-1, 4] * data_close)
                last = 'buy'
                last_prices = data[-1, 4] 
                last_data_close = data_close
                # print('p-p', (preds_next_time-preds_next_period).tolist())
                # print(' buy, pnl=', pnl*1e4, j)
        ###################### END STRATEGY #######################
        else: # 强制平仓
            if last == 'buy':
                TRADES += 1
                # 强制卖出
                pnl = (data[-1, 4] * data_close - last_prices * last_data_close) / (last_prices * last_data_close)
                print('force: sell, pnl=', pnl*1e4)
                revenue += (data[-1, 4] * data_close - last_prices * last_data_close)
            elif last == 'sell':
                TRADES += 1
                # 强制买入
                pnl = (last_prices * last_data_close - data[-1, 4] * data_close) / (last_prices * last_data_close)
                print('force: buy, pnl=', pnl*1e4)
                revenue += (last_prices * last_data_close - data[-1, 4] * data_close)

        if pnl != 0:
            PNL += pnl
            PNLS = np.append(PNLS, pnl.cpu().detach().numpy())
        # print(PNLS)
        if pnl > 0:
            WINS += 1

    if (i+1) % 10 == 0 and TRADES:
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
print('revenue:', PNL*TRADES)
