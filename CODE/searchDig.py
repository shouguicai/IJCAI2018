# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from multiprocessing import Process
import multiprocessing
import time

def searchDig(data):
    # 每个时间点之前用户的搜索次数
    print('search_cumsum_ing')
    user_search = data.groupby(['user_id','context_timestamp'],as_index=False)['context_page_id'].count()
    user_search['search_cumsum'] = user_search.groupby(['user_id'])['context_page_id'].cumsum()
    del user_search['context_page_id']

    data = pd.merge(data,user_search,on = ['user_id','context_timestamp'],how='left')
    #在timestamp之前用户搜索某类商品的次数
    print('predict_category_property_ing')
    for i in range(4):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:int(str(x.split(";")[i]).split(":")[0]) if len(x.split(";")) > i else -1
        )
        user_search_cat = data.groupby(['user_id','predict_category_%d'%(i),'context_timestamp'],as_index=False)['context_page_id'].count()
        data['search_category_%d_cumsum'%(i)] = user_search_cat.groupby(['user_id','predict_category_%d'%(i)])['context_page_id'].cumsum()
        del data['predict_category_%d'%(i)]

    del data['predict_category_property']
    del data['context_page_id']
    #data = pd.merge(data,user_search_cat,on=['user_id','context_timestamp'],how='left')
    print('user_mean_age_ing')
    #每个商品用户年龄等级均值
    user_mean_age = data.groupby(['item_id'],as_index=False)['user_age_level'].mean().rename(columns={'user_age_level':'user_age_item_mean'})
    data = pd.merge(data,user_mean_age,on=['item_id'],how='left')
    return data


if __name__=='__main__':
    save_path='data/feature/searchDig.csv' 
    train = pd.read_csv("data/origin/round2_train.csv")
    test = pd.read_csv("data/origin/round2_ijcai_18_test_b_20180510.csv")
    data = pd.concat([train, test])
    origin_features = data.columns.values.tolist()
    data = data.drop_duplicates(subset=['instance_id','item_id','user_id','context_id','shop_id'])  # 按instance id去重
    data = searchDig(data)
    exist_features = data.columns.values.tolist()
    ret_list = list(set(origin_features)&set(exist_features))
    for c in ret_list:
        if c not in ['instance_id','item_id','user_id','context_id','shop_id']:
            del data[c]
    data.to_csv(save_path,index=False) 
    print('search Dig 完成')