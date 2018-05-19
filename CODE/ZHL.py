# -*- coding: utf-8 -*-

# 提取转化率，包括商品，店铺，品牌，城市，用户

from multiprocessing import Process
import multiprocessing
import time
import scipy.special as special
from collections import Counter
import pandas as pd 

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha) < epsilon and abs(new_beta-self.beta) < epsilon:
                break
            #if i%(iter_num//20) ==0 :
            #    print(new_alpha,new_beta,i//iter_num)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

# 商品 item_id
def PHItem():  
    df_item = pd.read_csv('data/origin/item_id.csv') 
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    item_all_list = list(set(df_item.item_id.values)) 
    del df_item    
    print('PHItem: 数据读取完成,开始统计转化率')

    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.item_id.values)) # 统计训练数据中所有商品广告点击次数,保存为map形式:<item_id,count>
    dic_cov=dict(Counter(df_train[df_train['is_trade']==1].item_id.values))  # 统计训练数据中成功交易的商品广告点击次数,保存为map形式:<item_id,count>
    l=list(set(df_train.item_id.values))     
    I=[] # I 是广告点击次数
    C=[] # C 是成功交易次数
    for itemID in l:
        I.append(dic_i[itemID])
    for itemID in l:
        if itemID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[itemID])        
    print('PHItem: 开始做贝叶斯平滑')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('PHItem: 构建平滑转化率')
    dic_PH={}
    for item in item_all_list:
        if item not in dic_i:
            dic_PH[item]=(bs.alpha)/(bs.alpha+bs.beta)
        elif item not in dic_cov:
            dic_PH[item]=(bs.alpha)/(dic_i[item]+bs.alpha+bs.beta)
        else:
            dic_PH[item]=(dic_cov[item]+bs.alpha)/(dic_i[item]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
                         'PH_item':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PH_item.csv',index=False)

    print('PHItem: 保存完成')

# 店铺 shop_id
def PHShop():    
    df_shop = pd.read_csv('data/origin/shop_id.csv') 
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    shop_all_list = list(set(df_shop.shop_id.values)) 
    del df_shop    
    print('PHShop: 数据读取完成,开始统计转化率')

    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.shop_id.values)) # 统计训练数据中所有商品广告点击次数,保存为map形式:<item_id,count>
    dic_cov=dict(Counter(df_train[df_train['is_trade']==1].shop_id.values))  # 统计训练数据中成功交易的商品广告点击次数,保存为map形式:<item_id,count>
    l=list(set(df_train.shop_id.values))     
    I=[] # I 是广告点击次数
    C=[] # C 是成功交易次数
    for shopID in l:
        I.append(dic_i[shopID])
    for shopID in l:
        if shopID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[shopID])        
    print('PHShop: 开始做贝叶斯平滑')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('PHShop: 构建平滑转化率')
    dic_PH={}
    for shop in shop_all_list:
        if shop not in dic_i:
            dic_PH[shop]=(bs.alpha)/(bs.alpha+bs.beta)
        elif shop not in dic_cov:
            dic_PH[shop]=(bs.alpha)/(dic_i[shop]+bs.alpha+bs.beta)
        else:
            dic_PH[shop]=(dic_cov[shop]+bs.alpha)/(dic_i[shop]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'shop_id':list(dic_PH.keys()),
                         'PH_shop':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PH_shop.csv',index=False)

    print('PHShop: 保存完成')

# 品牌 item_brand_id
def PHBrand():    
    df_brand = pd.read_csv('data/origin/item_brand_id.csv') 
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    brand_all_list = list(set(df_brand.item_brand_id.values)) 
    del df_brand  
    print('PHBrand: 数据读取完成,开始统计转化率')

    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.item_brand_id.values)) # 统计训练数据中所有商品广告点击次数,保存为map形式:<item_id,count>
    dic_cov=dict(Counter(df_train[df_train['is_trade']==1].item_brand_id.values))  # 统计训练数据中成功交易的商品广告点击次数,保存为map形式:<item_id,count>
    l=list(set(df_train.item_brand_id.values))     
    I=[] # I 是广告点击次数
    C=[] # C 是成功交易次数
    for brandID in l:
        I.append(dic_i[brandID])
    for brandID in l:
        if brandID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[brandID])        
    print('PHBrand: 开始做贝叶斯平滑')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('PHBrand: 构建平滑转化率')
    dic_PH={}
    for brand in brand_all_list:
        if brand not in dic_i:
            dic_PH[brand]=(bs.alpha)/(bs.alpha+bs.beta)
        elif brand not in dic_cov:
            dic_PH[brand]=(bs.alpha)/(dic_i[brand]+bs.alpha+bs.beta)
        else:
            dic_PH[brand]=(dic_cov[brand]+bs.alpha)/(dic_i[brand]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'item_brand_id':list(dic_PH.keys()),
                         'PH_brand':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PH_brand.csv',index=False)

    print('PHBrand: 保存完成')

# 城市 item_city_id
def PHCity():    
    df_city = pd.read_csv('data/origin/item_city_id.csv') 
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    city_all_list = list(set(df_city.item_city_id.values)) 
    del df_city  
    print('PHCity: 数据读取完成,开始统计转化率')

    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.item_city_id.values)) # 统计训练数据中所有商品广告点击次数,保存为map形式:<item_id,count>
    dic_cov=dict(Counter(df_train[df_train['is_trade']==1].item_city_id.values))  # 统计训练数据中成功交易的商品广告点击次数,保存为map形式:<item_id,count>
    l=list(set(df_train.item_city_id.values))     
    I=[] # I 是广告点击次数
    C=[] # C 是成功交易次数
    for cityID in l:
        I.append(dic_i[cityID])
    for cityID in l:
        if cityID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[cityID])        
    print('PHCity: 开始做贝叶斯平滑')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('PHCity: 构建平滑转化率')
    dic_PH={}
    for city in city_all_list:
        if city not in dic_i:
            dic_PH[city]=(bs.alpha)/(bs.alpha+bs.beta)
        elif city not in dic_cov:
            dic_PH[city]=(bs.alpha)/(dic_i[city]+bs.alpha+bs.beta)
        else:
            dic_PH[city]=(dic_cov[city]+bs.alpha)/(dic_i[city]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'item_city_id':list(dic_PH.keys()),
                         'PH_city':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PH_city.csv',index=False)

    print('PHCity: 保存完成')

# 用户 user_id
def PHUser():    
    df_user = pd.read_csv('data/origin/user_id.csv') 
    df_train = pd.read_csv('data/origin/round2_train.csv')   
    user_all_list = list(set(df_user.user_id.values)) 
    del df_user  
    print('PHUser: 数据读取完成,开始统计转化率')

    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.user_id.values)) # 统计训练数据中所有商品广告点击次数,保存为map形式:<item_id,count>
    dic_cov=dict(Counter(df_train[df_train['is_trade']==1].user_id.values))  # 统计训练数据中成功交易的商品广告点击次数,保存为map形式:<item_id,count>
    l=list(set(df_train.user_id.values))     
    I=[] # I 是广告点击次数
    C=[] # C 是成功交易次数
    for userID in l:
        I.append(dic_i[userID])
    for userID in l:
        if userID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[userID])        
    print('PHUser: 开始做贝叶斯平滑')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('PHUser: 构建平滑转化率')
    dic_PH={}
    for user in user_all_list:
        if user not in dic_i:
            dic_PH[user]=(bs.alpha)/(bs.alpha+bs.beta)
        elif user not in dic_cov:
            dic_PH[user]=(bs.alpha)/(dic_i[user]+bs.alpha+bs.beta)
        else:
            dic_PH[user]=(dic_cov[user]+bs.alpha)/(dic_i[user]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'user_id':list(dic_PH.keys()),
                         'PH_user':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PH_user.csv',index=False)

    print('PHUser: 保存完成')

if __name__=='__main__':

    p1 = Process(target=PHItem)
    p2 = Process(target=PHShop)
    p3 = Process(target=PHBrand)
    p4 = Process(target=PHCity)
    p5 = Process(target=PHUser)

    start = time.time()

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    end = time.time()
    print('5 processes take %s seconds' % (end - start))