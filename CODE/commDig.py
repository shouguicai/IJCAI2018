# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from multiprocessing import Process
import multiprocessing
import time


def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def get_count_feat(all_data,data,long=3):
    end_time = data['context_timestamp'].min()
    begin_time = pd.to_datetime(end_time) - datetime.timedelta(days=long)
    all_data['context_timestamp'] = pd.to_datetime(all_data['context_timestamp'])
    all_data = all_data[
        (all_data['context_timestamp']<end_time)&(all_data['context_timestamp']>=begin_time)
                    ]
    print(end_time)
    print(begin_time)
    print(all_data['context_timestamp'].max()-all_data['context_timestamp'].min())
    item_count = all_data.groupby(['item_id'],as_index=False).size().reset_index()
    item_count.rename(columns={0:'item_count'},inplace=True)

    user_count = all_data.groupby(['user_id'], as_index=False).size().reset_index()
    user_count.rename(columns={0: 'user_count'}, inplace=True)
    return user_count,item_count

def pre_process(data):

    print('预处理')
    print('item_category_list_ing')
    for i in range(4):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:int(x.split(";")[i]) if len(x.split(";")) > i else -1
        )
    del data['item_category_list']

    print('item_property_list_ing')
    for i in range(4):
        data['property_%d'%(i)] = data['item_property_list'].apply(
            lambda x:int(x.split(";")[i]) if len(x.split(";")) > i else -1
        )
    del data['item_property_list']

    print('predict_category_property_ing_0')
    for i in range(4):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:int(str(x.split(";")[i]).split(":")[0]) if len(x.split(";")) > i else -1
        )
    del data['predict_category_property']
    
    # 时间化为24小时
    #df_datetime = data['context_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    #data['context_timehour'] = df_datetime.apply(lambda x: round(x.hour + 1.0*x.minute/60))

    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    print('context_timestamp_ing')
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
    data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])
    data['week'] = data['context_timestamp_tmp'].dt.weekday
    del data['context_timestamp_tmp']
    del data['time']
    del data['day']

    return data

def commDig(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_comm.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_comm.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_comm.csv'
    else:
        print('data_type出错！')
        return
    
    df_output = pre_process(df_output)

    all_data = pd.read_csv('data/origin/round2_train.csv')

    all_data = pre_process(all_data)

    user_count,item_count = get_count_feat(all_data,df_output,2)

    df_output = pd.merge(df_output,user_count,on=['user_id'],how='left')

    df_output = pd.merge(df_output,item_count,on=['item_id'],how='left')


    #df_output=df_output.drop(['item_id','item_category_list','item_property_list','item_brand_id','item_city_id','user_id','context_id','context_timestamp','predict_category_property','shop_id'],axis=1)

    # 数值归到个位
    # df_output=df_output.drop(['item_category_list','item_property_list','user_id','context_id','predict_category_property'],axis=1)

    #columns_name = df_output.columns.values.tolist()

    '''
    df_output['user_age_level'] = df_output['user_age_level'].apply(lambda x: -1 if x < 0 else x - 1000)

    df_output['user_occupation_id'] = df_output['user_occupation_id'].apply(lambda x: -1 if x < 0 else x - 2000)

    df_output['user_star_level'] = df_output['user_star_level'].apply(lambda x: -1 if x < 0 else x - 3000)

    df_output['context_page_id'] = df_output['context_page_id'].apply(lambda x: -1 if x < 0 else x - 4000)

    df_output['shop_star_level'] = df_output['shop_star_level'].apply(lambda x: -1 if x < 0 else x - 5000)
    '''
    '''
    # 时间化为24小时
    df_datetime = df_output['context_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    df_output['context_timehour'] = df_datetime.apply(lambda x: round(x.hour + 1.0*x.minute/60))
    '''
    # 5个id用于唯一标识一个样本，以免后面合并的时候出错
    # 16 个特征
    # 最后是标记 is_trade
    '''
    if data_type=='test':
        column_names = ['instance_id','item_id','user_id','context_id','shop_id',   
                        'item_price_level','item_collected_level','item_collected_level','item_pv_level',
                        'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                        'context_timehour','context_page_id','shop_review_num_level','shop_review_positive_rate',
                        'shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
    else:
        column_names = ['instance_id','item_id','user_id','context_id','shop_id',   
                        'item_price_level','item_collected_level','item_collected_level','item_pv_level',
                        'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                        'context_timehour','context_page_id','shop_review_num_level','shop_review_positive_rate',
                        'shop_star_level','shop_score_service','shop_score_delivery','shop_score_description',
                        'is_trade']

    df_output.to_csv(save_path,columns = column_names,index=False) 
    '''
    #del df_output['context_id']
    #del df_output['context_timestamp']
    df_output.to_csv(save_path,index=False) 
    
    print('Common Dig 完成')

if __name__=='__main__':
    #p1 = Process(target=commDig, args=('train',))
    #p2 = Process(target=commDig, args=('validate',))
    p3 = Process(target=commDig, args=('test',))
    start = time.time()
    #p1.start()
    #p2.start()
    p3.start()
    #p1.join()
    #p2.join()
    p3.join()
    end = time.time()
    print('3 processes take %s seconds' % (end - start))