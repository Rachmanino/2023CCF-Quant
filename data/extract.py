#######################每天使用前修改date#######################
date = '20231129'
###############################################################
import pandas as pd
import os
# 遍历~/wutong/data中的每一个csv文件
# 读取csv文件中的每一行，将每一行的第一列作为key，第二列作为value，存入字典
# 将字典写入csv文件
# 重复以上步骤，直到遍历完所有csv文件
csv_names = os.listdir('../../wutong/data')
cnt = 0
print('date:', date)
os.mkdir(date)
for csv_name in csv_names:
    csv_path = '../../wutong/data/' + csv_name
    rf = pd.read_csv(csv_path)
    a = rf.values.tolist()
    # print(a)
    b = []
    for i in range(len(a)):
        if a[i][-2] == int(date) and 93000 <= a[i][-1] <= 113100:   # filter
            b.append(a[i] + [0])
    wf = pd.DataFrame(b)
    wf.columns = ['open_price', 'high_price', 'low_price', 'close_price', 'vwap', 'money', 'volume', 'date', 'times', 'code_index']
    wf.to_csv(date + '/' + csv_name, index=True, mode='w')
    cnt += 1
    if cnt % 100 == 0:
        print(f'  {cnt} / {len(csv_names)}')
print("done")