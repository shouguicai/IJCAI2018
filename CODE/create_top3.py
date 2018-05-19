
# -*- coding: utf-8 -*-
# 分析热卖商品，品牌，归属城市，商家

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_top(num=3):
    df_output=pd.read_csv('data/origin/round2_train.csv')
    save_path='data/origin/top3.csv'
    columns_name =df_output.columns.values.tolist()
    # top3 item_id 分析
    df_data = df_output.loc[(df_output["is_trade"]==1),columns_name]
    encoder = LabelEncoder()
    item_id_cat = df_data["item_id"]
    item_id_cat_encoded = encoder.fit_transform(item_id_cat)

    arr=list(item_id_cat_encoded)
    arr_appear_dict = dict((a, arr.count(a)) for a in arr)
    arr_appear = arr_appear_dict.items()
    arr_appear_sorted = sorted(arr_appear, key=lambda x: x[1], reverse=True)

    item_top3 = ''
    for i in range(num):
        item_top3 += (str(encoder.classes_[arr_appear_sorted[i][0]])+' ')

    # top3 shop_id 分析
    encoder = LabelEncoder()
    shop_id_cat = df_data["shop_id"]
    shop_id_cat_encoded = encoder.fit_transform(shop_id_cat)

    arr=list(shop_id_cat_encoded)
    arr_appear_dict = dict((a, arr.count(a)) for a in arr)
    arr_appear = arr_appear_dict.items()
    arr_appear_sorted = sorted(arr_appear, key=lambda x: x[1], reverse=True)

    shop_top3 = ''
    for i in range(num):
        shop_top3 += (str(encoder.classes_[arr_appear_sorted[i][0]])+' ')

    # top3 item_city_id 分析
    encoder = LabelEncoder()
    item_city_cat = df_data["item_city_id"]
    item_city_cat_encoded = encoder.fit_transform(item_city_cat)

    arr=list(item_city_cat_encoded)
    arr_appear_dict = dict((a, arr.count(a)) for a in arr)
    arr_appear = arr_appear_dict.items()
    arr_appear_sorted = sorted(arr_appear, key=lambda x: x[1], reverse=True)

    city_top3 = ''
    for i in range(num):
        city_top3 += (str(encoder.classes_[arr_appear_sorted[i][0]])+' ')

    # top3 item_brand_id 分析
    encoder = LabelEncoder()
    item_brand_cat = df_data["item_brand_id"]
    item_brand_cat_encoded = encoder.fit_transform(item_brand_cat)

    arr=list(item_brand_cat_encoded)
    arr_appear_dict = dict((a, arr.count(a)) for a in arr)
    arr_appear = arr_appear_dict.items()
    arr_appear_sorted = sorted(arr_appear, key=lambda x: x[1], reverse=True)

    brand_top3 = ''
    for i in range(num):
        brand_top3 += (str(encoder.classes_[arr_appear_sorted[i][0]])+' ')

    print(item_top3)
    print(shop_top3)
    print(city_top3)
    print(brand_top3)

    data={'item_top3':item_top3,
           'shop_top3':shop_top3,
           'city_top3':city_top3,
           'brand_top3':brand_top3} 
    df_list=pd.DataFrame(data,index=[0])


    column_names = ['item_top3','shop_top3','city_top3','brand_top3']
    
    df_list.to_csv(save_path,columns=column_names,index=False)

    print('分析完成')

if __name__=='__main__':
    create_top()
