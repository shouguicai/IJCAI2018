# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

import time


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def map_hour(x):
    if (x>=7)&(x<=12):
        return 1
    elif (x>=13)&(x<=20):
        return 2
    else:
        return 3

def deliver(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=4.1+jiange*(i-1))&(x<=4.1+jiange*i):
            return i+1
    if x==-5:
        return 1

def deliver1(x):
    if (x>=2)&(x<=4):
        return 1
    elif (x>=5)&(x<=7):
        return 2
    else:
        return 3


def review(x):
    # x=round(x,6)
    jiange = 0.02
    for i in range(1, 30):
        if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
            return i + 1
    if x == -1:
        return 1

def review1(x):
    # x=round(x,6)
    if (x>=2)&(x<=12):
        return 1
    elif (x>=13)&(x<=15):
        return 2
    else:
        return 3


def service(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def service1(x):
    if (x>=2)&(x<=7):
        return 1
    elif (x>=8)&(x<=9):
        return 2
    else:
        return 3


def describe(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,30):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def describe1(x):
    if (x>=2)&(x<=8):
        return 1
    elif (x>=9)&(x<=10):
        return 2
    else:
        return 3

def shop_fenduan(data):
    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)
    # del data['shop_score_delivery']
    print(data.deliver_map.value_counts())

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)
    # del data['shop_score_service']
    print(data.service_map.value_counts())  # 视为好评，中评，差评
    #
    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)
    # del data['shop_score_description']
    print(data.de_map.value_counts())

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)
    print(data.review_map.value_counts())

    data['normal_shop'] = data.apply(
        lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
        axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data


def slide_cnt(data):
    # item_cnt = data.groupby(by='item_id').count()['instance_id'].to_dict()
    # data['item_cnt'] = data['item_id'].apply(lambda x: item_cnt[x])
    # user_cnt = data.groupby(by='user_id').count()['instance_id'].to_dict()
    # data['user_cnt'] = data['user_id'].apply(lambda x: user_cnt[x])
    # shop_cnt = data.groupby(by='shop_id').count()['instance_id'].to_dict()
    # data['shop_cnt'] = data['shop_id'].apply(lambda x: shop_cnt[x])
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day

    print('当前日期前一天的cnt')
    for d in range(1, 7):  # 18到24号
        df1 = data[data['day'] == d - 1]
        df2 = data[data['day'] == d]  # 19到25号
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 1:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    print('当前日期之前的cnt')
    for d in range(1, 7):
        # 19到25，25是test
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 1:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    del data['realtime']
    del data['day']
    print("前一个小时的统计量")

    return data


def zuhe(data):
    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item两两组合')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user两两组合')
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop两两组合')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']


    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',  'sale_price','sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level','gender_age','gender_occ','gender_star',
                'shop_review_num_level','shop_star_level','review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data

def item(data):
    print('一个item有多少brand,price salse collected level……')

    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')

    for col in ['item_brand_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg({str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob']=data[str(col) + '_item_cnt']/data['item_cnt']
    del data['item_cnt']

    print('一个brand有多少price salse collected level……')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg({'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg({str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
    del data['item_brand_cnt']

    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
    del data['item_city_cnt']

    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']

    print('一个item_collected_level有多少item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg({'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg({str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
    del data['item_coll_cnt']

    return data

def user(data):
    print('用户有多少性别')
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    for col in ['user_gender_id','user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob']=data[str(col) + '_user_cnt']/data['user_cnt']
    del data['user_cnt']

    print('性别的年龄段，职业有多少')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob']=data[str(col) + '_user_gender_cnt']/data['user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level对应的user_occupation_id，user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob']=data[str(col) + '_user_age_cnt']/data['user_age_cnt']
    del data['user_age_cnt']

    print('user_occupation_id对应的user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob']=data[str(col) + '_user_occ_cnt']/data['user_occ_cnt']
    del data['user_occ_cnt']

    return data

def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('一个user有多少item_id,item_brand_id……')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col)+'_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col)+'_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col)+'_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']

    return data


def user_shop(data):
    print('一个user有多少shop_id,shop_review_num_level……')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']

    return data


def shop_item(data):
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
    del data['shop_cnt']

    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg({'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)['instance_id'].agg({str(col)+'_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
    del data['shop_rev_cnt']

    return data

if __name__ == "__main__":
    save_path='data/feature/snowDig.csv' 
    train = pd.read_csv("data/origin/round2_train.csv")
    test = pd.read_csv("data/origin/round2_ijcai_18_test_b_20180510.csv")
    data = pd.concat([train, test])
    origin_features = data.columns.values.tolist()
    data = data.drop_duplicates(subset='instance_id')  # 按instance id去重
    print('make feature')
    data=shop_fenduan(data)
    data = slide_cnt(data)
    data = zuhe(data)
    print('----------------------------全局统计特征---------------------------------------------------')
    data = item(data)
    data = user(data)
    data = user_item(data)
    data = user_shop(data)
    data = shop_item(data)
    for c in origin_features:
        if c != 'instance_id':
            del data[c]
    data.to_csv(save_path,index=False) 