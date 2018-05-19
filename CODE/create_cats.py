# -*- coding: utf-8 -*-

# 根据训练数据和测试数据，构建商品，店铺，品牌，城市，用户的目录文件
# 供后续求转化率以及做平滑用
import pandas as pd

def create_cat():
    df_train = pd.read_csv('data/origin/round2_train.csv')
    df_test = pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')

    print('数据读取完成，开始统计')

    # 商品 item_id
    item_set_on_train = set(df_train['item_id']) 
    item_set_on_test = set(df_test['item_id']) 
    item_set_all = item_set_on_train | item_set_on_test # 求并集
    item_id_list = list(item_set_all)
    df_list=pd.DataFrame({'item_id':item_id_list})
    df_list.to_csv('data/origin/item_id.csv',index=False)

    # 店铺 shop_id
    shop_set_on_train = set(df_train['shop_id']) 
    shop_set_on_test = set(df_test['shop_id']) 
    shop_set_all = shop_set_on_train | shop_set_on_test # 求并集
    shop_id_list = list(shop_set_all)
    df_list=pd.DataFrame({'shop_id':shop_id_list})
    df_list.to_csv('data/origin/shop_id.csv',index=False)

    # 品牌 item_brand_id
    item_brand_set_on_train = set(df_train['item_brand_id']) 
    item_brand_set_on_test = set(df_test['item_brand_id']) 
    item_brand_set_all = item_brand_set_on_train | item_brand_set_on_test # 求并集
    item_brand_id_list = list(item_brand_set_all)
    df_list=pd.DataFrame({'item_brand_id':item_brand_id_list})
    df_list.to_csv('data/origin/item_brand_id.csv',index=False)

    # 城市 item_city_id
    item_city_set_on_train = set(df_train['item_city_id']) 
    item_city_set_on_test = set(df_test['item_city_id']) 
    item_city_set_all = item_city_set_on_train | item_city_set_on_test # 求并集
    item_city_id_list = list(item_city_set_all)
    df_list=pd.DataFrame({'item_city_id':item_city_id_list})
    df_list.to_csv('data/origin/item_city_id.csv',index=False)

    # 用户 user_id
    user_set_on_train = set(df_train['user_id']) 
    user_set_on_test = set(df_test['user_id']) 
    user_set_all = user_set_on_train | user_set_on_test # 求并集
    user_id_list = list(user_set_all)
    df_list=pd.DataFrame({'user_id':user_id_list})
    df_list.to_csv('data/origin/user_id.csv',index=False)

    print('目录统计完成')

if __name__=='__main__':
    create_cat()