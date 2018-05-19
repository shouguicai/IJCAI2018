# -*- coding: utf-8 -*-

import time
import datetime
import pandas as pd

def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def sel_train():
    df_merge = pd.read_csv('data/cutData/train/train_20180425_merge.csv')
    save_path = 'data/cutData/train/train_20180425_merge_sel.csv'
    print('原始数据读取完成')
    columns_name = ['instance_id','item_id','user_id','context_id','shop_id']

    df_sel = df_merge[df_merge['context_timestamp']>'2018-09-03 23:59:59']
    print('选择完成') 
    df_sel.to_csv(save_path,index=False) 
    print('保存完成')
    del df_sel
    del df_merge

if __name__=='__main__':
    sel_train()