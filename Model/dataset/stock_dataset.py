import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

SCALE_MONEY = 1e9

SCALE_VOLUME = 1e6

class Stock_Dataset(Dataset):
    def __init__(self, data_dir,day_windows=30, flag='train', device ="cpu", scale_money =SCALE_MONEY, scale_volume= SCALE_VOLUME) -> None:
        # assert flag in ['train', 'valid', 'test']
        self.flag = flag
        self.day_windows = day_windows
        self.device =device

        '''定义需要使用到的指标，可以自己添加或者选择'''
        INDICATORS = [
            "close_price",
            "open_price",
            "high_price",
            "low_price",
            'vwap',
            'money',
            'volume',
            'code_index'
        ]

        self.indicator_num = len(INDICATORS)
        data_dir = f"{data_dir}/{flag}.csv"
        self.data_np = convert_dataframe2numpy(data_dir, INDICATORS)

        self.scale_money = scale_money
        self.scale_volume = scale_volume

        self.predict_period = 10




    def __len__(self):
        return len(self.data_np) - 2*self.day_windows

    def __getitem__(self, index:int):
        ###[date_windows, stock_num, indicator]
        data = self.data_np[index:index+self.day_windows].copy()

        next_data_close = self.data_np[index+self.day_windows, 4].copy()

        next_period = self.data_np[index+self.day_windows: index+self.day_windows+self.predict_period, 4].copy().mean()

        data_close = data[-1,0]

        # reward_ratio = (next_data_close - data_close) / data_close
        reward_ratio = (next_data_close - data_close) / data_close
        next_period = (next_period - data_close) / data_close

        ####所有的价格都除以最后一天的收盘价,可以把数据价格统一归一化到1附近
        data[:,:5] = data[:,:5] / data_close.reshape((1,-1,1))     ####np.repeat(last_close, self.indicator_num, axis=2)

        ####换手率不需要归一化，因为一般都在1附近左右  成交量需要归一化  除以统一的缩放因子即可
        data[:,5] = data[:,5] / self.scale_money
        data[:,6] = data[:,6] / self.scale_volume

        '''归一化处理  可以将所有数据在对应的维度进行归一化'''

        data = torch.as_tensor(data, dtype=torch.float32).to(self.device)
        target = torch.as_tensor([reward_ratio, next_period], dtype=torch.float32).reshape(2,).to(self.device)

        ###转换为百分比的涨幅
        target = target * 100
        # print('data_close', data_close)
        return data, target, data_close


def convert_dataframe2numpy(data_path, indicator):
    data_df = pd.read_csv(data_path)
    data_df = data_df.set_index(data_df.columns[0])
    # stock_num = len(data_df.Name.unique())
    array = data_df[indicator].to_numpy().reshape(-1,
                                                  len(indicator))  ###[date, indicator]
    return array




# """test for dataset"""
# ds = Stock_Dataset(data_dir ="./data",day_windows=30, flag='train', device="cuda:0")
# data, y = ds.__getitem__(0)
#
# print(data)
# print(y)