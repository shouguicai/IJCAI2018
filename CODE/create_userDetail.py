# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
  
#把原数据转化成一个用户一条记录的形式
#train + test data 一起做
def create_UserAction():
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    df_test =  pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
    df_test['is_trade'] = -1 # 表示未知    
    df = pd.concat([df_train,df_test])
    del df_train
    del df_test
    #记录总共的记录条数
    total_num=len(df)
    #获取所有的userID
    userID_list=df['user_id'].values
    userID_list=list(set(userID_list))
    dict_item_id ={}
    dict_item_category_list  ={}
    dict_item_property_list  ={}
    dict_item_brand_id ={}
    dict_item_city_id ={} 
    dict_item_price_level ={} 
    dict_item_sales_level ={}  
    dict_item_collected_level ={}   
    dict_item_pv_level ={} 
    dict_user_gender_id ={} 
    dict_user_age_level ={}
    dict_user_occupation_id ={}  
    dict_user_star_level ={}
    dict_context_timestamp ={}  
    dict_context_page_id ={}
    dict_predict_category_property ={} 
    dict_shop_id ={}
    dict_shop_review_num_level ={}  
    dict_shop_review_positive_rate ={} 
    dict_shop_star_level ={}
    dict_shop_score_service ={} 
    dict_shop_score_delivery ={}
    dict_shop_score_description ={} 
    dict_is_trade ={}
    dict_clickTime={}
    dict_trade_his={}
    #初始化字典
    for user in userID_list:
        dict_item_id[user] =''
        dict_item_category_list[user] =''
        dict_item_property_list[user] =''
        dict_item_brand_id[user] =''
        dict_item_city_id[user] =''
        dict_item_price_level[user] =''
        dict_item_sales_level[user] =''
        dict_item_collected_level[user] =''  
        dict_item_pv_level[user] =''
        dict_user_gender_id[user] =''
        dict_user_age_level[user] =''
        dict_user_occupation_id[user] =''
        dict_user_star_level[user] =''
        dict_context_timestamp[user] =''
        dict_context_page_id[user] =''
        dict_predict_category_property[user] ='' 
        dict_shop_id[user] =''
        dict_shop_review_num_level[user] =''
        dict_shop_review_positive_rate[user] =''
        dict_shop_star_level[user] =''
        dict_shop_score_service[user] =''
        dict_shop_score_delivery[user] =''
        dict_shop_score_description[user] =''
        dict_is_trade[user] = -1
        dict_clickTime[user]=''
        dict_trade_his[user]=''
    print('开始整理和统计每个用户记录')
    count=0
    for A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA in df.values:
        dict_item_id[K] +=(str(B)+' ')
        dict_item_category_list[K] +=(str(C)+' ')
        dict_item_property_list[K] +=(str(D)+' ')
        dict_item_brand_id[K] +=(str(E)+' ')
        dict_item_city_id[K] +=(str(F)+' ')
        dict_item_price_level[K] +=(str(G)+' ')
        dict_item_sales_level[K] +=(str(H)+' ')
        dict_item_collected_level[K] +=(str(I)+' ')
        dict_item_pv_level[K] +=(str(J)+' ')
        if dict_user_gender_id[K] =='':
            dict_clickTime[K]=1
            dict_user_gender_id[K] =str(L)
            dict_user_age_level[K] =str(M)
            dict_user_occupation_id[K] =str(N)
            dict_user_star_level[K] =str(O)
        else:
            dict_clickTime[K]+=1
        dict_context_timestamp[K] +=(str(Q)+' ')
        dict_context_page_id[K] +=(str(R)+' ')
        dict_predict_category_property[K] +=(str(S)+' ')
        dict_shop_id[K] +=(str(T)+' ')
        dict_shop_review_num_level[K] +=(str(U)+' ')
        dict_shop_review_positive_rate[K] +=(str(V)+' ')
        dict_shop_star_level[K] +=(str(W)+' ')
        dict_shop_score_service[K] +=(str(X)+' ')
        dict_shop_score_delivery[K] +=(str(Y)+' ')
        dict_shop_score_description[K] +=(str(Z)+' ')
        if dict_is_trade[K] != 1 and AA != -1:
            dict_is_trade[K] = AA
        dict_trade_his[K] +=(str(AA)+' ')

        count+=1
        if count%50000==0:
            print('字典已加载: %.2f %%'%(1.0*count/total_num*100))
    print('整理完成，开始保存')
    
    #保存
    item_id_list=list(dict_item_id.values())
    item_category_list=list(dict_item_category_list.values())  
    item_property_list=list(dict_item_property_list.values())  
    item_brand_id=list(dict_item_brand_id.values())
    item_city_id=list(dict_item_city_id.values())    
    item_price_level=list(dict_item_price_level.values())    
    item_sales_level=list(dict_item_sales_level.values())
    item_collected_level=list(dict_item_collected_level.values())    
    item_pv_level=list(dict_item_pv_level.values())   
    user_gender_id=list(dict_user_gender_id.values())  
    user_age_level=list(dict_user_age_level.values())  
    user_occupation_id=list(dict_user_occupation_id.values())  
    user_star_level=list(dict_user_star_level.values()) 
    context_timestamp =list(dict_context_timestamp.values())  
    context_page_id =list(dict_context_page_id.values())
    predict_category_property =list(dict_predict_category_property.values())  
    shop_id = list(dict_shop_id.values())  
    shop_review_num_level =list(dict_shop_review_num_level.values())  
    shop_review_positive_rate =list(dict_shop_review_positive_rate.values())
    shop_star_level=list(dict_shop_star_level.values()) 
    shop_score_service=list(dict_shop_score_service.values())  
    shop_score_delivery=list(dict_shop_score_delivery.values()) 
    shop_score_description=list(dict_shop_score_description.values())  
    is_trade=list(dict_is_trade.values())
    trade_his = list(dict_trade_his.values())
    clickTime=list(dict_clickTime.values())

    
    df_list=pd.DataFrame({'user_id':userID_list,
                        'item_id_list' :item_id_list,
                        'item_category_list' :item_category_list,
                        'item_property_list' : item_property_list,
                        'item_brand_id_list' :item_brand_id,
                        'item_city_id_list': item_city_id,
                        'item_price_level_list' : item_price_level,
                        'item_sales_level_list' :item_sales_level,
                        'item_collected_level_list' : item_collected_level,
                        'item_pv_level_list' :item_pv_level,
                        'user_gender_id': user_gender_id,
                        'user_age_level' :user_age_level,
                        'user_occupation_id': user_occupation_id,
                        'user_star_level': user_star_level,
                        'context_timestamp_list' :context_timestamp,
                        'context_page_id_list' :context_page_id,
                        'predict_category_property_list' :predict_category_property,
                        'shop_id_list' :shop_id,
                        'shop_review_num_level_list' :shop_review_num_level,
                        'shop_review_positive_rate_list' : shop_review_positive_rate,
                        'shop_star_level_list' :shop_star_level,
                        'shop_score_service_list' : shop_score_service ,
                        'shop_score_delivery_list': shop_score_delivery,
                        'shop_score_description_list' :shop_score_description,
                        'is_trade' :is_trade,
                        'trade_his':trade_his,
                        'clickTime' :clickTime})
    
    columns_name = ['user_id',
                    'item_id_list',
                    'item_category_list',
                    'item_property_list',
                    'item_brand_id_list' ,
                    'item_city_id_list',
                    'item_price_level_list',
                    'item_sales_level_list',
                    'item_collected_level_list',
                    'item_pv_level_list',
                    'user_gender_id_list',
                    'user_age_level',
                    'user_occupation_id',
                    'user_star_level',
                    'context_timestamp_list',
                    'context_page_id_list',
                    'predict_category_property_list',
                    'shop_id_list',
                    'shop_review_num_level_list',
                    'shop_review_positive_rate_list',
                    'shop_star_level_list',
                    'shop_score_service_list',
                    'shop_score_delivery_list',
                    'shop_score_description_list',
                    'is_trade',
                    'trade_his',
                    'clickTime']
    df_list.to_csv('data/feature/user_detail.csv',columns=columns_name,index=False)
    
if __name__=='__main__':
    create_UserAction()