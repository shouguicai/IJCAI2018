# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from multiprocessing import Process
import multiprocessing
import time

def Top3Dig(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_top3.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_top3.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_top3.csv'
    else:
        print('data_type出错！')
        return

    df_top3 = pd.read_csv('data/origin/top3.csv')

    item_top3=[int(i) for i in str(df_top3['item_top3'].values[0]).strip().split(' ')]
    shop_top3=[int(i) for i in str(df_top3['shop_top3'].values[0]).strip().split(' ')]
    city_top3=[int(i) for i in str(df_top3['city_top3'].values[0]).strip().split(' ')]
    brand_top3=[int(i) for i in str(df_top3['brand_top3'].values[0]).strip().split(' ')]

    # top3 item_id 分析

    df_output['item_top3'] = df_output["item_id"].apply(lambda x: 1 if int(x) in item_top3 else 0)

    # top3 shop_id 分析

    df_output['shop_top3'] = df_output["shop_id"].apply(lambda x: 1 if int(x) in shop_top3 else 0)

    # top3 item_city_id 分析

    df_output['city_top3'] = df_output["item_city_id"].apply(lambda x: 1 if int(x) in city_top3 else 0)

    # top3 item_brand_id 分析

    df_output['brand_top3'] = df_output["item_brand_id"].apply(lambda x: 1 if int(x) in brand_top3 else 0)

    print('处理完成')

    columns_name = ['instance_id','item_id','user_id','context_id','shop_id',   
                    'item_top3','shop_top3','city_top3','brand_top3']

    df_output.to_csv(save_path,columns=columns_name,index=False) 

    print('保存完成')

if __name__=='__main__':
    
    p1 = Process(target=Top3Dig, args=('train',))
    p2 = Process(target=Top3Dig, args=('validate',))
    p3 = Process(target=Top3Dig, args=('test',))
    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    end = time.time()
    print('3 processes take %s seconds' % (end - start))
    