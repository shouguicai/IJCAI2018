# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import time

def time2cov(time_):

    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def cutTrain(day=1):
    df_ori = pd.read_csv('data/origin/round2_train.csv')
    df = df_ori.copy()
    print('原始数据读取完成')
    columns_name = df.columns.values.tolist()
    df['time2cov'] = df['context_timestamp'].apply(time2cov)
    ''' 
    # old version
    df_datetime = df['context_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_day = df_datetime.apply(lambda x: x.day)

    df_out = df.loc[(df_day == day),columns_name]
    df_out_1 = df.loc[(df_day != day),columns_name]
    '''
    df_out = df[df['time2cov']>'2018-09-05 23:59:59']
    df_out_1 = df[df['time2cov']<='2018-09-05 23:59:59']
    del df_out['time2cov']
    del df_out_1['time2cov']
    print('切割完成，开始保存') 
    df_out.to_csv('data/cutData/validate/train_time_validate.csv',index=False)
    df_out_1.to_csv('data/cutData/train/train_time_data.csv',index=False)
    print('保存完成')
    del df
    del df_out
    del df_out_1

if __name__=='__main__':
    cutTrain(day=1)