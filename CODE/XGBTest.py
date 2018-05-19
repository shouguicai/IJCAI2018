# -*- coding: utf-8 -*-

import time
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg') # 没有GUI时使用matplotlib绘图,写在import pylab前
from matplotlib import pylab as plt

#XGB的模型训练代码

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split


def loadCSV(path=''):
    return pd.read_csv(path)

def ceate_feature_map(features):
    outfile = open('model/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def get_importance(bst):

  importance = bst.get_fscore(fmap='model/xgb.fmap')
  importance = sorted(importance.items(), key=operator.itemgetter(1))

  df = pd.DataFrame(importance, columns=['feature', 'fscore'])
  df['fscore'] = df['fscore'] / df['fscore'].sum()
  df.to_csv('model/feature_importance.csv',index=False)
  '''
  plt.figure()
  df.plot()
  df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
  plt.title('XGBoost Feature Importance')
  plt.xlabel('relative importance')
  plt.gcf().savefig('model/feature_importance_xgb_331.png')
  '''

#train = loadCSV('data/cutData/train/train_20180425_merge.csv') 

val = loadCSV('data/cutData/validate/validate_20180425_merge.csv') 

test = loadCSV('data/cutData/test/test_20180425_merge.csv')

#train = train.fillna(-1)

val = val.fillna(-1)

test = test.fillna(-1)

#y_train = train.pop('is_trade')

y_val = val.pop('is_trade')

ins_test = test.pop('instance_id') 

#train.pop('instance_id') 

val.pop('instance_id') 

#sel = ['item_sales_level_user_occ_prob', 'item_city_id_user_prob', 'item_id_user_gender_cnt', 'category_2', 'item_cnt1', 'predict_category_2', 'item_top3', 'item_sales_level_item_prob', 'item_pv_level_user_cnt', 'item_collected_level_salse_prob', 'shop_review_num_level_user_gender_prob', 'item_id_shop_cnt', 'user_occupation_id_user_cnt', 'brand_top3', 'item_id_shop_rev_cnt', 'item_id_user_occ_cnt', 'item_collected_level_user_occ_prob', 'search_cumsum', 'item_pv_level_item_prob', 'item_collected_level_shop_prob', 'n_count_item', 'n_click_cur_brand', 'item_price_level_user_occ_prob', 'shop_id_user_prob', 'normal_shop', 'sale_collect', 'item_price_level_city_prob', 'item_collected_level_price_prob', 'hour', 'user_query_day', 'user_gender_id_user_cnt', 'item_price_level_shop_rev_prob', 'n_cur_item', 'item_brand_id_user_occ_prob', 'user_age_level', 'user_age_level_user_gender_prob', 'item_sales_level', 'item_price_level_user_cnt', 'shop_star_level_user_prob', 'item_pv_level_brand_prob', 'item_price_level_user_gender_prob', 'n_count_shop', 'category_0', 'category_3', 'is_convered_item', 'item_id_user_gender_prob', 'item_id_user_occ_prob', 'item_id_user_age_prob', 'is_convered_shop', 'user_occupation_id_user_prob', 'user_star_level_user_age_prob', 'item_collected_level_shop_rev_prob', 'item_city_id_item_prob', 'shop_id_user_gender_prob', 'shop_id_user_age_prob', 'shop_id_user_occ_prob', 'n_click_cur_item', 'user_age_level_user_prob', 'item_pv_level_shop_rev_prob', 'n_click_cur_shop', 'shop_review_positive_rate', 'item_collected_level_city_prob', 'is_convered_brand', 'user_star_level_user_occ_prob', 'shop_score_service', 'item_id_shop_rev_prob', 'item_sales_level_city_prob', 'shop_score_delivery', 'item_brand_id_item_prob', 'item_city_id_user_occ_prob', 'shop_score_description', 'user_star_level_user_prob']

#train = train[sel]

#val = val[sel]

#test = test[sel]

feature_name = val.columns.values.tolist()

del_features = ['realtime','day','context_id'] # ,'search_category_1_cumsum'

feat_import = loadCSV('model/dif_on_feature_mean_4_25.csv')

#feat_import = loadCSV('model/feature_importance_XGBFS.csv')

#feat_import = loadCSV('model/feature_importance_new.csv')

# 9 36 67
del_features += list(feat_import['feature'].values[:43])

del_features = list(set(del_features))

for c in del_features:
  if c in feature_name:
    #del train[c]
    del val[c]
    del test[c]
    
#del train['context_timestamp']

#del val['context_timestamp']

print('CSV加载完成')


'''
# V1
X_train = train.drop(['item_id','user_id','context_id','shop_id',
                      'category_0','category_1','category_2','property_0',
                      'property_1','property_2','predict_category_0',
                      'predict_category_1','predict_category_2'],axis=1).values

X_val = val.drop(['item_id','user_id','context_id','shop_id',
                      'category_0','category_1','category_2','property_0',
                      'property_1','property_2','predict_category_0',
                      'predict_category_1','predict_category_2'],axis=1).values

X_test = test.drop(['item_id','user_id','context_id','shop_id',
                      'category_0','category_1','category_2','property_0',
                      'property_1','property_2','predict_category_0',
                      'predict_category_1','predict_category_2'],axis=1).values
'''


# V2
'''
enc = OneHotEncoder()
lb = LabelEncoder()
feat_set = feature_name
for i,feat in enumerate(feat_set):
    print(feat)
    tmp = lb.fit_transform((list(train[feat])+list(val[feat])+list(test[feat])))
    enc.fit(tmp.reshape(-1,1))
    x_train = enc.transform(lb.transform(list(train[feat])).reshape(-1, 1))
    x_test = enc.transform(lb.transform(list(test[feat])).reshape(-1, 1))
    x_val = enc.transform(lb.transform(list(val[feat])).reshape(-1, 1))
    if i == 0:
        X_train,X_val,X_test = x_train,x_val,x_test
    else:
        X_train,X_val,X_test = sparse.hstack((X_train, x_train)),sparse.hstack((X_val, x_val)),sparse.hstack((X_test, x_test))
'''

# V3
lb = LabelEncoder()
feat_set = ['item_id','user_id','shop_id','item_brand_id','item_city_id','category_0','category_1','category_2','property_0','property_1','property_2','predict_category_0','predict_category_1','predict_category_2','predict_category_3']
for feat in feat_set:
    if feat not in del_features and feat in sel:
      print(feat)
      #lb.fit_transform((list(train[feat])+list(val[feat])+list(test[feat])))
      lb.fit_transform((list(val[feat])+list(test[feat])))
      #train[feat] = lb.transform(list(train[feat]))
      test[feat] = lb.transform(list(test[feat]))
      val[feat] = lb.transform(list(val[feat]))

#X_train = train.values

#X_val = val.values
X_test = test.values

#xgb_val = xgb.DMatrix(X_val,label=y_val)

#xgb_train = xgb.DMatrix(X_train, label=y_train)

xgb_test = xgb.DMatrix(X_test)

del test

del X_test

#del X_val

#del X_train

params={
#'booster':'gbtree',
'objective': 'binary:logistic', #多分类的问题
'gamma':0,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':7, # 构建树的深度，越大越容易过拟合
'lambda':80,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'scale_pos_weight':1, # 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
'min_child_weight':2, #这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.07, # 如同学习率,通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2
'seed':1080,
#'nthread':7,# cpu 线程数
'eval_metric': 'logloss'
}

plst = list(params.items())


num_rounds = 2000 # 迭代次数

#watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

online = True

# training model 
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
#model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)

#bst = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=20)
'''
del xgb_val

del xgb_train

best_iter = bst.best_iteration
'''
best_iter = 302

if online == True:

  print('训练模型')

  # V2
  #X_train = sparse.vstack([X_train,X_val])

  # V3
  '''
  X_train = pd.concat([train,val],ignore_index=True).values

  y_train = pd.concat([y_train,y_val],ignore_index=True)
  '''
  #del train

  X_train = val

  y_train = y_val

  del val

  xgb_train = xgb.DMatrix(X_train, label=y_train)

  watchlist = [(xgb_train, 'train')]

  model = xgb.train(plst, xgb_train, best_iter, watchlist)

  '''
  ceate_feature_map(feature_name)
   
  get_importance(model)

  print('特征重要性计算完成')
  '''
  prob = model.predict(xgb_test)

  output = pd.DataFrame({'instance_id':ins_test,'predicted_score':prob})

  initial_ins = loadCSV('result/initial_instance_id.csv') 

  output = pd.merge(initial_ins,output,how='left',on='instance_id')

  output.to_csv('result/submission_xgb_5_15.txt', index=False, sep=' ', columns={'instance_id', 'predicted_score'})

  print('结果预测完成')
