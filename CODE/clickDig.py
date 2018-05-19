# -*- coding: utf-8 -*-

'''
 统计特征
 提取用户点击行为有关的特征
 包括四个条目：商品，店铺，品牌，城市
 每个条目包含：点击总数，点击当前广告的总数 n_count_,n_cur_
               当前时刻以前的点击总数，当前时刻以前的点击当前广告的总数 n_click_count_,n_click_cur_
               以前是否已经买过该商品/该店/该品牌/该城市 is_convered_
'''
import numpy as np
import pandas as pd 
from multiprocessing import Process
import multiprocessing
import time

def clickDetail(x):

    n_count_shop = 0
    n_count_city = 0
    n_count_brand = 0
    n_count_item = 0

    n_cur_shop = 0
    n_cur_city = 0
    n_cur_brand = 0
    n_cur_item = 0

    n_click_count_shop = 0
    n_click_count_city = 0
    n_click_count_brand = 0
    n_click_count_item = 0

    n_click_cur_shop = 0
    n_click_cur_city = 0
    n_click_cur_brand = 0
    n_click_cur_item = 0

    is_convered_shop = 0
    is_convered_city = 0
    is_convered_brand = 0
    is_convered_item = 0

    cur_time=int(str(x.context_timestamp))
    
    if not isinstance(x.context_timestamp_list,float):
        time_list=[int(i) for i in x.context_timestamp_list.strip().split(' ')]
        shop_list=[int(i) for i in x.shop_id_list.strip().split(' ')]
        city_list=[int(i) for i in x.item_city_id_list.strip().split(' ')]
        brand_list=[int(i) for i in x.item_brand_id_list.strip().split(' ')]
        item_list=[int(i) for i in x.item_id_list.strip().split(' ')]
        trade_his_list = [int(i) for i in x.trade_his.strip().split(' ')]

        #print(time_list)
        #print(shop_list)
        #print(city_list)
        #print(brand_list)
        #print(item_list)
        #print(trade_his_list)

        for i in range(len(time_list)):
            n_count_shop+=1
            n_count_city+=1
            n_count_brand+=1
            n_count_item+=1

            if x.shop_id==shop_list[i]:
                n_cur_shop+=1   

            if x.item_city_id==city_list[i]:
                n_cur_city+=1

            if x.item_brand_id==brand_list[i]:
                n_cur_brand+=1     

            if x.item_id==item_list[i]:
                n_cur_item+=1                                     
                
            if time_list[i] < cur_time:
                n_click_count_shop  +=1
                n_click_count_city +=1
                n_click_count_brand +=1
                n_click_count_item +=1

                if x.shop_id==shop_list[i]:
                    n_click_cur_shop+=1   

                if x.item_city_id==city_list[i]:
                    n_click_cur_city+=1

                if x.item_brand_id==brand_list[i]:
                    n_click_cur_brand+=1     

                if x.item_id==item_list[i]:
                    n_click_cur_item+=1   

                if is_convered_shop==0:                  
                    if shop_list[i]==x.shop_id and trade_his_list[i]==1:
                        is_convered_shop=1       

                if is_convered_city==0:                  
                    if city_list[i]==x.item_city_id and trade_his_list[i]==1:
                        is_convered_city=1   

                if is_convered_brand==0:                  
                    if brand_list[i]==x.item_brand_id and trade_his_list[i]==1:
                        is_convered_brand=1   

                if is_convered_item==0:                  
                    if item_list[i]==x.item_id and trade_his_list[i]==1:
                        is_convered_item=1          
    
    re=str(n_count_shop)+','+str(n_count_city)+','+str(n_count_brand)+','+str(n_count_item)+','+ \
       str(n_cur_shop)+','+str(n_cur_city)+','+str(n_cur_brand)+','+str(n_count_item)+','+ \
       str(n_click_count_shop)+','+str(n_click_count_city)+','+str(n_click_count_brand)+','+str(n_click_count_item)+','+ \
       str(n_click_cur_shop)+','+str(n_click_cur_city)+','+str(n_click_cur_brand)+','+str(n_click_cur_item)+','+\
       str(is_convered_shop)+','+str(is_convered_city)+','+str(is_convered_brand)+','+str(is_convered_item)
    
    # print(re)
    
    return re
    
#挖掘统计信息
def getn_count_shop(x):
    return x.split(',')[0]

def getn_count_city(x):
    return x.split(',')[1]

def getn_count_brand(x):
    return x.split(',')[2]

def getn_count_item(x):
    return x.split(',')[3]

def getn_cur_shop(x):
    return x.split(',')[4]

def getn_cur_city(x):
    return x.split(',')[5]

def getn_cur_brand(x):
    return x.split(',')[6]

def getn_cur_item(x):
    return x.split(',')[7]

def getn_click_count_shop(x):
    return x.split(',')[8]

def getn_click_count_city(x):
    return x.split(',')[9]

def getn_click_count_brand(x):
    return x.split(',')[10]

def getn_click_count_item(x):
    return x.split(',')[11]

def getn_click_cur_shop(x):
    return x.split(',')[12]

def getn_click_cur_city(x):
    return x.split(',')[13]

def getn_click_cur_brand(x):
    return x.split(',')[14]

def getn_click_cur_item(x):
    return x.split(',')[15]

def getis_convered_shop(x):
    return x.split(',')[16]

def getis_convered_city(x):
    return x.split(',')[17]

def getis_convered_brand(x):
    return x.split(',')[18]

def getis_convered_item(x):
    return x.split(',')[19]

def digClickDetail(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_click.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_click.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_click.csv'       
    else:
        print('data_type出错！')
        return
        
    df_detail=pd.read_csv('data/feature/user_detail.csv')

    print('数据读取完成')

    df_user_action = df_detail[['user_id','item_id_list','item_brand_id_list','item_city_id_list','shop_id_list','context_timestamp_list','trade_his']]
    
    df_output=pd.merge(df_output,df_user_action,how='left',on='user_id')
    
    print('拼接完成,开始统计')
    
    del df_detail
        
    df_output['click_Detail']=df_output.apply(lambda x:clickDetail(x),axis=1)
    
    
    df_output['n_count_shop']=df_output['click_Detail'].apply(lambda x:getn_count_shop(x))
    df_output['n_count_city']=df_output['click_Detail'].apply(lambda x:getn_count_city(x))
    df_output['n_count_brand']=df_output['click_Detail'].apply(lambda x:getn_count_brand(x))
    df_output['n_count_item']=df_output['click_Detail'].apply(lambda x:getn_count_item(x))

    df_output['n_cur_shop']=df_output['click_Detail'].apply(lambda x:getn_cur_shop(x))
    df_output['n_cur_city']=df_output['click_Detail'].apply(lambda x:getn_cur_city(x))
    df_output['n_cur_brand']=df_output['click_Detail'].apply(lambda x:getn_cur_brand(x))
    df_output['n_cur_item']=df_output['click_Detail'].apply(lambda x:getn_cur_item(x))

    df_output['n_click_count_shop']=df_output['click_Detail'].apply(lambda x:getn_click_count_shop(x))
    df_output['n_click_count_city']=df_output['click_Detail'].apply(lambda x:getn_click_count_city(x))
    df_output['n_click_count_brand']=df_output['click_Detail'].apply(lambda x:getn_click_count_brand(x))
    df_output['n_click_count_item']=df_output['click_Detail'].apply(lambda x:getn_click_count_item(x))

    df_output['n_click_cur_shop']=df_output['click_Detail'].apply(lambda x:getn_click_cur_shop(x))
    df_output['n_click_cur_city']=df_output['click_Detail'].apply(lambda x:getn_click_cur_city(x))
    df_output['n_click_cur_brand']=df_output['click_Detail'].apply(lambda x:getn_click_cur_brand(x))
    df_output['n_click_cur_item']=df_output['click_Detail'].apply(lambda x:getn_click_cur_item(x))

    df_output['is_convered_shop']=df_output['click_Detail'].apply(lambda x:getis_convered_shop(x))
    df_output['is_convered_city']=df_output['click_Detail'].apply(lambda x:getis_convered_city(x))
    df_output['is_convered_brand']=df_output['click_Detail'].apply(lambda x:getis_convered_brand(x))
    df_output['is_convered_item']=df_output['click_Detail'].apply(lambda x:getis_convered_item(x))


    print('点击细节挖掘完成')  

    # df_output=df_output.drop(['click_Detail','item_id_list','item_brand_id_list','item_city_id_list','shop_id_list','context_timestamp_list'],axis=1)
    
    # 5个id用于唯一标识一个样本，以免后面合并的时候出错
    columns_name = ['instance_id','item_id','user_id','context_id','shop_id',   
                    'n_count_shop','n_count_city','n_count_brand','n_count_item',
                    'n_cur_shop','n_cur_city','n_cur_brand','n_cur_item',
					'n_click_count_shop','n_click_count_city','n_click_count_brand','n_click_count_item',
					'n_click_cur_shop','n_click_cur_city','n_click_cur_brand','n_click_cur_item',
                    'is_convered_shop','is_convered_city','is_convered_brand','is_convered_item']

    df_output.to_csv(save_path,columns=columns_name,index=False) 

    print('保存完成')
    
if __name__=='__main__':

    p1 = Process(target=digClickDetail, args=('train',))
    p2 = Process(target=digClickDetail, args=('validate',))
    p3 = Process(target=digClickDetail, args=('test',))
    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    end = time.time()
    print('3 processes take %s seconds' % (end - start))
    