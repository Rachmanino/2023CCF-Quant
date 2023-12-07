import os, sys
from tkinter.tix import MAX

import pandas as pd
sys.path.append("..")
from quote_api.quote_api import QuoteApi
from proto import message_pb2
from proto import type_pb2
from proto.type_pb2 import SDSDataType
import time
from time import sleep
import datetime
import torch
import numpy as np
import pickle
import csv

class QuoteImpl(QuoteApi):
    def __init__(self, ip, port):
        QuoteApi.__init__(self, ip, port)
        self.login_success_ = False
        self.user = ''
        self.pwd = ''
        self.mac = ''
        self.kline_data = dict()

        ##################### 初始化策略中每只股票需要的信息 ###################
        self.last = dict()  # 记录上次买or卖
        self.last_prices = dict() # 记录上次交易的价格，-1表示待下一分钟确定
        self.cnt = dict() # 记录交易次数

        # 数据转存，方便回测
        csv_names = os.listdir('../../../wutong/data')
        for csv_name in csv_names:
            stock_name = csv_name[:-4]
            self.last[stock_name] = None
            self.last_prices[stock_name] = 100
            self.cnt[stock_name] = 0
        
    def setUserInfo(self,username,password):
         self.user = username
         self.pwd = password


    def isLogin(self):
        return self.login_success_

    def isConnected(self):
        return self.client.isConnected()


    def onMsg(self,msg):
        try:
            head = msg.head
            if head.msg_type == type_pb2.kMtLoginMegateAns:
                self.onLoginAns(msg)
            elif head.msg_type == type_pb2.kMtPublish:
                self.onPublish(msg)
        except Exception as e:
            print(e)

    def onTick(self,tick):
        '''
        # default strategy
        # print(tick)
        self.kilne_queue(tick)
        if len(self.kline_data[tick['hjcode']]) >2:
            if tick['close_price'] > tick['vwap']:

                self.sell(tick['hjcode'])
            else:
                self.buy(tick['hjcode'])
        '''
        #####run the test data on the model 

        # parameters 
        THRESHOLDS = 0.01 
        MAXTIME = 10000 # 最大连续上涨或下跌次数
        
        # 数据预处理的参数
        scale_money = 1e9
        scale_volume = 1e6
        predict_period = 10
        
        # 导入Model和Config
        model_save_dir = "../../Model/valid/loadmodule/model_end.pt"
        config_save_dir = "../../Model/valid/loadmodule/config.obj"
        
        # load config
        with open(config_save_dir, 'rb') as f:
            config = pickle.load(f)    

        # load model
        model = torch.load(model_save_dir, map_location=config.device)

        # check whether the data is enough
        self.kilne_queue(tick)
        if len(self.kline_data[tick['hjcode']]) < 30:
            return

        # load data
        data = []
        for i in range(len(self.kline_data[tick['hjcode']])):
            code_index = 0  # 我们的模型里没有用到core_index，所以这里全取0
            data.append( [  self.kline_data[tick['hjcode']][i][3],         # close_price
                            self.kline_data[tick['hjcode']][i][0],         # open_price 
                            self.kline_data[tick['hjcode']][i][1],         # high_price
                            self.kline_data[tick['hjcode']][i][2],         # low_price
                            self.kline_data[tick['hjcode']][i][4],         # vwap
                            self.kline_data[tick['hjcode']][i][5],         # money
                            self.kline_data[tick['hjcode']][i][6],         # volume
                            code_index] )
        data = np.array(data)
        data_close = data[-1,0]
        data[:,:5] = data[:,:5] / data_close.reshape((1,-1,1)) 
        # 所有的价格都除以最后一天的收盘价,可以把数据价格统一归一化到1附近

        # 换手率不需要归一化，因为一般都在1附近左右  
        # 成交量需要归一化  除以统一的缩放因子即可
        data[:,5] = data[:,5] / scale_money
        data[:,6] = data[:,6] / scale_volume
        data = torch.as_tensor(data, dtype=torch.float32).to(config.device)

        # run on our model
        with torch.no_grad():
            model.eval()
            preds = model(data.unsqueeze(0))[0]   
            preds_next_time = preds[0]  # 预测下一分钟的涨跌幅
            preds_next_period = preds[1]    # 预测下一时间段（10分钟）的平均涨跌幅

        ####################### our strategy #######################
        # 1.确定上一分钟交易的价格
        if self.last_prices[tick['hjcode']] == -1:
            self.last_prices[tick['hjcode']] = data[-1][4]
            # print(f"last f{self.last[tick['hjcode']]} price: {self.last_prices[tick['hjcode']]}")

        # 2.判断是否需要交易
        self.cnt[tick['hjcode']] += 1
        if self.last[tick['hjcode']] != 'sell' and \
                ((preds_next_time - preds_next_period > THRESHOLDS and data[-1,0] * (1.0 + preds_next_time/100) > self.last_prices[tick['hjcode']])
                 or self.cnt[tick['hjcode']] >= MAXTIME):
            self.sell(tick['hjcode'])
            self.cnt[tick['hjcode']] = 0
            self.last_prices[tick['hjcode']] = -1 #下一分钟才知道价格
            self.last[tick['hjcode']] = 'sell'
            with open('sell.txt', 'a') as f:
                f.write(f"p-p = {preds_next_time - preds_next_period}, last = {self.last[tick['hjcode']]} sell\n")
        elif self.last[tick['hjcode']] != 'buy' and \
                ((preds_next_period - preds_next_time > THRESHOLDS and data[-1,0] * (1.0 + preds_next_time/100) < self.last_prices[tick['hjcode']])
                 or self.cnt[tick['hjcode']] >= MAXTIME):
            self.buy(tick['hjcode'])
            self.cnt[tick['hjcode']] = 0
            self.last_prices[tick['hjcode']] = -1 #下一分钟才知道价格
            self.last[tick['hjcode']] = 'buy'
            with open('buy.txt', 'a') as f:
                f.write(f"p-p = {preds_next_time - preds_next_period}, last = {self.last[tick['hjcode']]} buy\n")
        #############################################################
        ''''
        print('last:', self.last[tick['hjcode']])
        print('last_prices:', self.last_prices[tick['hjcode']])
        print('cnt:', self.cnt[tick['hjcode']])
        '''

    def onPublish(self,msg):
        """订阅数据推送
        """
        for msg in self.message_buffer[msg.head.topic]:
            if(msg.head.topic == SDSDataType.KLN):
                kln = message_pb2.SDSKLine()
                kln.ParseFromString(msg.data)
                kline = {}
                timestamp = kln.timems / 1000
                dt = datetime.datetime.fromtimestamp(timestamp)
                kline['hjcode'] = kln.hjcode
                kline['date'] = int(dt.strftime("%Y%m%d"))
                kline['times'] = int(dt.strftime("%H%M%S"))
                kline['open_price'] = kln.open*1.0/10000 
                kline['high_price'] = kln.high*1.0/10000 
                kline['low_price'] = kln.low*1.0/10000 
                kline['close_price'] = kln.last*1.0/10000
                if kln.volume == 0:
                    kline['vwap'] = kline['open_price']
                else:
                    kline['vwap'] = kln.turnover*1.0/kln.volume/10000
                kline['money'] = kln.turnover
                kline['volume'] = kln.volume
                self.onTick(kline)
                print("pub kln:",kln)

    def kilne_queue(self, kln):
        max_len = 30
        now_kln = [kln['open_price'], kln['high_price'], kln['low_price'], kln['close_price'], kln['vwap'],
                   kln['money'], kln['volume']]
        if kln['hjcode'] not in self.kline_data.keys():
            self.kline_data[kln['hjcode']] = []
        self.kline_data[kln['hjcode']].append(now_kln)
        if len(self.kline_data[kln['hjcode']]) > max_len:
            self.kline_data[kln['hjcode']].pop(0)
        print(self.kline_data[kln['hjcode']])
        # print('len = ', len(self.kline_data[kln['hjcode']]))

        ###################################### 交易数据转存 ######################################
        with open(f"../../../wutong/data/{kln['hjcode']}.csv", mode='a', newline='', encoding='utf8') as file:    # mode='a'表示追加
            writer = csv.writer(file)
            writer.writerow(now_kln + [kln['date'], kln['times']]) 

    def onConected(self, connected):
        if connected:
            print("connect success,try login")
            self.login(self.user, self.pwd, self.mac)
        else:
            print("connect loss")

    def onLoginAns(self, msg):
        """登录应答
        """
        login_ans = message_pb2.LoginAns()
        login_ans.ParseFromString(msg.data)
        if(login_ans.ret == login_ans.success):
            self.login_success_ = True
            print("login success")
        elif(login_ans.ret == login_ans.acc_error):
            print("acc error")
        elif(login_ans.ret == login_ans.not_login):
            print("账号已在别处登录！")
