# -*- coding: utf-8 -*-

'''
 基于时序的一些挖掘
'''
import numpy as np
import pandas as pd 
from multiprocessing import Process
import multiprocessing
import time

#构造组合标签字典
def createTimeDict():

    df_train=pd.read_csv('data/origin/round2_train.csv').drop(['is_trade'],axis=1)
    df_test=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
    
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
    return dic_count

def digTimeSpam(x,dic_data):

    timeSpam=-1 #保存与上条点击记录的时间间隔,-1代表没有间隔
    timeSpam_shop = -1 #保存与上条点击记录的时间间隔(同shop),-1代表没有间隔
    timeSpam_city =-1#保存与上条点击记录的时间间隔(同city),-1代表没有间隔
    timeSpam_brand = -1 #保存与上条点击记录的时间间隔(同brand),-1代表没有间隔
    timeSpam_item =-1#保存与上条点击记录的时间间隔(同item),-1代表没有间隔
    
    cur_time=int(str(x.context_timestamp))

    if not isinstance(x.context_timestamp,float):
        time_list=[int(i) for i in x.context_timestamp_list.strip().split(' ')]
        shop_list=[int(i) for i in x.shop_id_list.strip().split(' ')]
        city_list=[int(i) for i in x.item_city_id_list.strip().split(' ')]
        brand_list=[int(i) for i in x.item_brand_id_list.strip().split(' ')]
        item_list=[int(i) for i in x.item_id_list.strip().split(' ')]
                  
        lasttime=-1 #与当前点击最近的历史点击时间
        lasttime_shop=-1
        lasttime_city=-1
        lasttime_brand=-1 
        lasttime_item=-1 #与当前点击最近的历史点击同item时间

        for i in range(len(time_list)):
            #跳出
            if(time_list[i]>cur_time):
                break
            elif time_list[i]==cur_time:
                #当前是重复数据
                if x.isRepeat>0:
                    timeSpam=0
                    if x.shop_id==shop_list[i]:
                        timeSpam_shop=0
                    if x.item_city_id==city_list[i]:
                        timeSpam_city=0
                    if x.item_brand_id==brand_list[i]:
                        timeSpam_brand=0
                    if x.item_id==item_list[i]:
                        timeSpam_item=0
                break
            else:
                lasttime=time_list[i]
                if x.shop_id==shop_list[i]:
                    lasttime_shop=time_list[i]
                if x.item_city_id==city_list[i]:
                    lasttime_city =time_list[i]
                if x.item_brand_id==brand_list[i]:
                    lasttime_brand = time_list[i]
                if x.item_id==item_list[i]:
                    lasttime_item = time_list[i]
        if timeSpam!=0:
            if lasttime==-1:
                timeSpam=-1
            else:
                timeSpam=cur_time-lasttime
            
        if timeSpam_shop!=0:
            if lasttime_shop==-1:
                timeSpam_shop=-1
            else:
                timeSpam_shop=cur_time - lasttime_shop    

        if timeSpam_city!=0:
            if lasttime_city==-1:
                timeSpam_city=-1
            else:
                timeSpam_city=cur_time - lasttime_city

        if timeSpam_brand!=0:
            if lasttime_brand==-1:
                timeSpam_brand=-1
            else:
                timeSpam_brand=cur_time - lasttime_brand

        if timeSpam_item!=0:
            if lasttime_item==-1:
                timeSpam_item=-1
            else:
                timeSpam_item=cur_time - lasttime_item   

    ct_usr = dic_data.loc[dic_data['dict']==('ct'+str(cur_time)+','+'usr'+str(x.user_id)),['count']].values.tolist()[0][0]
    ct_shop = dic_data.loc[dic_data['dict']==('ct'+str(cur_time)+','+'shop'+str(x.shop_id)),['count']].values.tolist()[0][0]
    ct_city = dic_data.loc[dic_data['dict']==('ct'+str(cur_time)+','+'city'+str(x.item_city_id)),['count']].values.tolist()[0][0]
    ct_brand = dic_data.loc[dic_data['dict']==('ct'+str(cur_time)+','+'brand'+str(x.item_brand_id)),['count']].values.tolist()[0][0]
    ct_item = dic_data.loc[dic_data['dict']==('ct'+str(cur_time)+','+'item'+str(x.item_id)),['count']].values.tolist()[0][0]
    '''
    dic_count=createTimeDict()
    ct_usr=dic_count['ct'+str(cur_time)+','+'usr'+str(x.user_id)]
    ct_shop=dic_count['ct'+str(cur_time)+','+'shop'+str(x.shop_id)]
    ct_city=dic_count['ct'+str(cur_time)+','+'city'+str(x.item_city_id)]
    ct_brand=dic_count['ct'+str(cur_time)+','+'brand'+str(x.item_brand_id)]
    ct_item=dic_count['ct'+str(cur_time)+','+'item'+str(x.item_id)]
    '''
    re=str(timeSpam)+','+str(timeSpam_shop)+','+str(timeSpam_city)+','+str(timeSpam_brand)+','+str(timeSpam_item)
    re+=','+str(ct_usr)+','+str(ct_shop)+','+str(ct_city)+','+str(ct_brand)+','+str(ct_item)
    
    return re

def get_timeSpam(x):
    return x.split(',')[0]

def get_timeSpam_shop(x):
    return x.split(',')[1]

def get_timeSpam_city(x):
    return x.split(',')[2]

def get_timeSpam_brand(x):
    return x.split(',')[3]

def get_timeSpam_item(x):
    return x.split(',')[4]

def get_ct_usr(x):
    return x.split(',')[5]

def get_ct_shop(x):
    return x.split(',')[6]

def get_ct_city(x):
    return x.split(',')[7]

def get_ct_brand(x):
    return x.split(',')[8]

def get_ct_item(x):
    return x.split(',')[9]

def repeatFeature(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_time.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_time.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_time.csv'       
    else:
        print('data_type出错！')
        return
    feature_all=df_output.values.tolist()

    repeatList=[]
    total=len(feature_all)

    print('csv加载完成，开始计算') 
    flag=0
    for i in range(total):
        if flag==0:
            if i+1<total and feature_all[i]==feature_all[i+1]:
                flag=1
            repeatList.append(0)
        else:           
            repeatList.append(flag)
            flag+=1
            if i+1<total and feature_all[i]!=feature_all[i+1]:
                flag=0
            
        if i%5000==0:
            print('已统计 %.2f %%'%(1.0*i/total*100))
        
    del feature_all    
    
    df_output['isRepeat']=repeatList

    del repeatList

    df_detail=pd.read_csv('data/feature/user_detail.csv')

    df_user_action = df_detail[['user_id','item_id_list','item_brand_id_list','item_city_id_list','shop_id_list','context_timestamp_list']]

    df_all=pd.merge(df_output,df_user_action,how='left',on='user_id')

    del df_detail

    dic_data = pd.read_csv('data/feature/timeDict.csv')
    print('字典读取完成')

    df_all['timeDetail']=df_all.apply(lambda x:digTimeSpam(x,dic_data),axis=1)
    
    df_all['timeSpam']=df_all['timeDetail'].apply(lambda x:get_timeSpam(x))
    df_all['timeSpam_shop']=df_all['timeDetail'].apply(lambda x:get_timeSpam_shop(x))
    df_all['timeSpam_city']=df_all['timeDetail'].apply(lambda x:get_timeSpam_city(x))
    df_all['timeSpam_brand']=df_all['timeDetail'].apply(lambda x:get_timeSpam_brand(x))
    df_all['timeSpam_item']=df_all['timeDetail'].apply(lambda x:get_timeSpam_item(x))

    df_all['ct_usr']=df_all['timeDetail'].apply(lambda x:get_ct_usr(x))
    df_all['ct_shop']=df_all['timeDetail'].apply(lambda x:get_ct_shop(x))
    df_all['ct_city']=df_all['timeDetail'].apply(lambda x:get_ct_city(x))
    df_all['ct_brand']=df_all['timeDetail'].apply(lambda x:get_ct_brand(x))
    df_all['ct_item']=df_all['timeDetail'].apply(lambda x:get_ct_item(x))

    print('time细节挖掘完成')

    columns_name = ['instance_id','item_id','user_id','context_id','shop_id',   
                    'timeSpam','timeSpam_shop','timeSpam_city','timeSpam_brand','timeSpam_item',
                    'ct_usr','ct_shop','ct_city','ct_brand','ct_item']

    df_all.to_csv(save_path,columns=columns_name,index=False) 

    print('保存完成')


# 反向dig
def digTimeSpamReverse(x):
    
    timeSpamRe=-1      #保存与上条点击记录的时间间隔,-1代表没有间隔
    timeSpam_shopRe = -1 #保存与上条点击记录的时间间隔(同shop),-1代表没有间隔
    timeSpam_cityRe =-1#保存与上条点击记录的时间间隔(同city),-1代表没有间隔
    timeSpam_brandRe = -1 #保存与上条点击记录的时间间隔(同brand),-1代表没有间隔
    timeSpam_itemRe =-1 #保存与上条点击记录的时间间隔(同item),-1代表没有间隔
    
    cur_time=int(str(x.context_timestamp))

    if not isinstance(x.context_timestamp,float):
        time_list=[int(i) for i in x.context_timestamp_list.strip().split(' ')]
        shop_list=[int(i) for i in x.shop_id_list.strip().split(' ')]
        city_list=[int(i) for i in x.item_city_id_list.strip().split(' ')]
        brand_list=[int(i) for i in x.item_brand_id_list.strip().split(' ')]
        item_list=[int(i) for i in x.item_id_list.strip().split(' ')]
                  
        lasttimeRe=-1 #与当前点击最近的历史点击时间
        lasttime_shopRe=-1
        lasttime_cityRe=-1
        lasttime_brandRe=-1 
        lasttime_itemRe=-1 #与当前点击最近的历史点击同item时间

        time_list.reverse()
        shop_list.reverse()
        city_list.reverse()
        brand_list.reverse()
        item_list.reverse()

        for i in range(len(time_list)):
            #跳出
            if(time_list[i]<cur_time):
                break
            elif time_list[i]==cur_time:
                #当前是重复数据
                if x.isRepeatRe>0:
                    timeSpamRe=0
                    if x.shop_id==shop_list[i]:
                        timeSpam_shopRe=0
                    if x.item_city_id==city_list[i]:
                        timeSpam_cityRe=0
                    if x.item_brand_id==brand_list[i]:
                        timeSpam_brandRe=0
                    if x.item_id==item_list[i]:
                        timeSpam_itemRe=0
                break
            else:
                lasttimeRe=time_list[i]
                if x.shop_id==shop_list[i]:
                    lasttime_shopRe=time_list[i]
                if x.item_city_id==city_list[i]:
                    lasttime_cityRe =time_list[i]
                if x.item_brand_id==brand_list[i]:
                    lasttime_brandRe = time_list[i]
                if x.item_id==item_list[i]:
                    lasttime_itemRe = time_list[i]
        if timeSpamRe!=0:
            if lasttimeRe==-1:
                timeSpamRe=-1
            else:
                timeSpamRe=lasttimeRe - cur_time
            
        if timeSpam_shopRe!=0:
            if lasttime_shopRe==-1:
                timeSpam_shopRe=-1
            else:
                timeSpam_shopRe=lasttime_shopRe - cur_time    

        if timeSpam_cityRe!=0:
            if lasttime_cityRe==-1:
                timeSpam_cityRe=-1
            else:
                timeSpam_cityRe= lasttime_cityRe - cur_time 

        if timeSpam_brandRe!=0:
            if lasttime_brandRe==-1:
                timeSpam_brandRe=-1
            else:
                timeSpam_brandRe=lasttime_brandRe - cur_time

        if timeSpam_itemRe!=0:
            if lasttime_itemRe==-1:
                timeSpam_itemRe=-1
            else:
                timeSpam_itemRe= lasttime_itemRe - cur_time       
                
    re=str(timeSpamRe)+','+str(timeSpam_shopRe)+','+str(timeSpam_cityRe)+','+str(timeSpam_brandRe)+','+str(timeSpam_itemRe)
    
    return re

def get_timeSpamRe(x):
    return x.split(',')[0]

def get_timeSpam_shopRe(x):
    return x.split(',')[1]

def get_timeSpam_cityRe(x):
    return x.split(',')[2]

def get_timeSpam_brandRe(x):
    return x.split(',')[3]

def get_timeSpam_itemRe(x):
    return x.split(',')[4]


def repeatFeatureRe(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train/train_time_data.csv')
        save_path='data/cutData/train/train_20180425_reverse_time.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')
        save_path='data/cutData/test/test_20180425_reverse_time.csv'
    elif data_type=='validate':
        df_output=pd.read_csv('data/cutData/validate/train_time_validate.csv')
        save_path='data/cutData/validate/validate_20180425_reverse_time.csv'       
    else:
        print('data_type出错！')
        return
    feature_all=df_output.values.tolist()

    repeatListRe=[]
    total=len(feature_all)

    print('csv加载完成，开始计算') 
    flag=0
    for i in range(total):
        if flag==0:
            if i+1<total and feature_all[i]==feature_all[i+1]:
                flag=1
            repeatListRe.append(0)
        else:           
            repeatListRe.append(flag)
            flag+=1
            if i+1<total and feature_all[i]!=feature_all[i+1]:
                flag=0
            
        if i%5000==0:
            print('已统计 %.2f %%'%(1.0*i/total*100))
    repeatListRe.reverse()  
    del feature_all    
    
    df_output['isRepeatRe']=repeatListRe

    del repeatListRe

    df_detail=pd.read_csv('data/feature/user_detail.csv')

    df_user_action = df_detail[['user_id','item_id_list','item_brand_id_list','item_city_id_list','shop_id_list','context_timestamp_list']]

    df_all=pd.merge(df_output,df_user_action,how='left',on='user_id')

    del df_detail

    df_all['timeDetail']=df_all.apply(lambda x:digTimeSpamReverse(x),axis=1)
    
    df_all['timeSpamRe']=df_all['timeDetail'].apply(lambda x:get_timeSpamRe(x))
    df_all['timeSpam_shopRe']=df_all['timeDetail'].apply(lambda x:get_timeSpam_shopRe(x))
    df_all['timeSpam_cityRe']=df_all['timeDetail'].apply(lambda x:get_timeSpam_cityRe(x))
    df_all['timeSpam_brandRe']=df_all['timeDetail'].apply(lambda x:get_timeSpam_brandRe(x))
    df_all['timeSpam_itemRe']=df_all['timeDetail'].apply(lambda x:get_timeSpam_itemRe(x))


    print('反向time细节挖掘完成')

    columns_name = ['instance_id','item_id','user_id','context_id','shop_id',   
                    'timeSpamRe','timeSpam_shopRe','timeSpam_cityRe','timeSpam_brandRe','timeSpam_itemRe']

    df_all.to_csv(save_path,columns=columns_name,index=False) 

    print('保存完成')

if __name__=='__main__':

    #p1 = Process(target=repeatFeature, args=('train',))
    #p2 = Process(target=repeatFeature, args=('validate',))
    #p3 = Process(target=repeatFeature, args=('test',))
    p4 = Process(target=repeatFeatureRe, args=('train',))
    p5 = Process(target=repeatFeatureRe, args=('validate',))
    p6 = Process(target=repeatFeatureRe, args=('test',))
    start = time.time()
    #p1.start()
    #p2.start()
    #p3.start()
    p4.start()
    p5.start()
    p6.start()
    #p1.join()
    #p2.join()
    #p3.join()
    p4.join()
    p5.join()
    p6.join()
    end = time.time()
    print('6 processes take %s seconds' % (end - start))
    