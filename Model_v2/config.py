import os
import pickle
from random import random

import numpy as np
import torch

class Config:
    def  __init__(self, args):
        self.cwd = args.save_dir
        self.batch_size = args.batch_size
        self.day_windows = args.day_windows
        self.predict_period = args.predict_period


        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.drop = args.drop
        self.device = torch.device(f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and (args.gpu_id >= 0)) else "cpu")
        self.if_remove = True
        self.random_seed = 0

        '''Arguments for data'''
        self.data_dir = "./dataset/data"
        self.INDICATORS = [
            "close_price",
            "open_price",
            "high_price",
            "low_price",
            'vwap',
            'money',
            'volume'
        ]

        self.indicator_num = len(self.INDICATORS) - 1
        # self.stock_num = 29
        # self.alphas = 512


        '''Arguments for network'''
        '''hyper-parameters for TCN'''
        self.tcn_kernel_size = 3
        self.tcn_levels = 4
        self.tcn_hidden_dim = 256

        '''hyper-parameters for embedding'''
        self.wind1_embeddings = 10
        self.wind2_embeddings = 23
        self.company_embeddings = 7
        # '''hyper-parameters for LSTM'''
        # self.lstm_input_dim = 128
        # self.lstm_hidden_size = 128

        '''training'''
        self.clip_grad_norm = 10
        self.log_file = f"{self.cwd}/training.log"

        '''model save'''
        self.if_over_write = True
        self.save_gap = 20


    def init_before_training(self):

        ###设置随机数的种子
        # seed = self.random_seed
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # ra
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float32)

        '''set cwd (current working directory) for saving model'''
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.day_windows}_{self.learning_rate.__name__[5:]}_{self.random_seed}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format

    def save(self):
        file_path = f"{self.cwd}/config.obj"
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

