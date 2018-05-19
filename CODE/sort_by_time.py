# -*- coding: utf-8 -*-
# 按时间升序排序

import pandas as pd
from multiprocessing import Process
import multiprocessing
import time

def sort(data_type='train'):
    path =''
    if data_type=='train':
        path = 'data/origin/round2_train.csv'
    elif data_type=='test':
        path = 'data/origin/round2_ijcai_18_test_b_20180510.csv'
    else:
        print('data_type出错！')
        return
    df_output=pd.read_csv(path)
    print('原始数据读取完成')
    # 按时间升序排序
    df_output = df_output.sort(['context_timestamp'],ascending=True)
    print('排序完成，开始保存') 
    df_output.to_csv(path,index=False)
    print('保存完成')
    del df_output

if __name__=='__main__':
    p1 = Process(target=sort, args=('train',))
    p2 = Process(target=sort, args=('test',))
    start = time.time()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print('sort finished ... 2 processes take %s seconds' % (end - start))