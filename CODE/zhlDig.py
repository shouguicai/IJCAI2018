# -*- coding: utf-8 -*-

from multiprocessing import Process
import multiprocessing
import time
import pandas as pd 

def zhlDig(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_zhl.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_zhl.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_zhl.csv'
    else:
        print('data_type出错！')
        return
     
    df_cityPH = pd.read_csv('data/feature/PH_city.csv')
    df_output=pd.merge(df_output,df_cityPH,how='left',on='item_city_id')
    del df_cityPH

    df_brandPH = pd.read_csv('data/feature/PH_brand.csv')
    df_output=pd.merge(df_output,df_brandPH,how='left',on='item_brand_id')
    del df_brandPH

    df_shopPH = pd.read_csv('data/feature/PH_shop.csv')
    df_output=pd.merge(df_output,df_shopPH,how='left',on='shop_id')
    del df_shopPH

    df_itemPH = pd.read_csv('data/feature/PH_item.csv')
    df_output=pd.merge(df_output,df_itemPH,how='left',on='item_id')
    del df_itemPH   
    
    print('保存中......')

    columns_name = ['instance_id','item_id','user_id','context_id','shop_id',   
                    'PH_city','PH_brand','PH_shop','PH_item']

    df_output.to_csv(save_path,columns=columns_name,index=False) 


if __name__=='__main__':

    p1 = Process(target=zhlDig, args=('train',)) 
    p2 = Process(target=zhlDig, args=('validate',))
    p3 = Process(target=zhlDig, args=('test',))

    start = time.time()

    p1.start()
    p2.start()
    p3.start()


    p1.join()
    p2.join()
    p3.join()


    end = time.time()
    print('3 processes take %s seconds' % (end - start))