# -*- coding: utf-8 -*-

'''
 创建组合标签字典，供timeDig使用
'''
import numpy as np
import pandas as pd 

#构造组合标签字典
def createTimeDict():

    df_train=pd.read_csv('data/origin/round2_train.csv').drop(['is_trade'],axis=1)
    df_test=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
    save_path='data/feature/timeDict.csv'
    df_all=pd.concat([df_train,df_test])
    
    del df_train
    del df_test
    
    dic_count={}
    
    #ct+usr
    temp=df_all.groupby(['context_timestamp','user_id'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'usr'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+usr构造完成')
    
    #ct+item
    temp=df_all.groupby(['context_timestamp','item_id'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'item'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+item构造完成')
    
    #ct+city
    temp=df_all.groupby(['context_timestamp','item_city_id'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'city'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+city构造完成')
            
    #ct+brand
    temp=df_all.groupby(['context_timestamp','item_brand_id'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'brand'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+brand构造完成')

    #ct+shop
    temp=df_all.groupby(['context_timestamp','shop_id'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'shop'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+shop构造完成')

    print('字典构造完成')
    
    del df_all

    df_out=pd.DataFrame({'dict':list(dic_count.keys()),
                         'count':list(dic_count.values())})

    df_out.to_csv(save_path,index=False) 

    print('保存完成')

    return dic_count

if __name__=='__main__':
	createTimeDict()