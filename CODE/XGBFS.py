# -*- coding: utf-8 -*-

# 后向特征删除

import operator
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import log_loss

from numpy import sort
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib as mpl
mpl.use('Agg') # 没有GUI时使用matplotlib绘图,写在import pylab前
from matplotlib import pylab as plt

def loadCSV(path=''):
    return pd.read_csv(path)

train = loadCSV('data/cutData/train/train_20180301_merge.csv') 

val = loadCSV('data/cutData/validate/validate_20180301_merge.csv') 

test = loadCSV('data/cutData/test/test_20180301_merge.csv')

train = train.fillna(-1)

val = val.fillna(-1)

test = test.fillna(-1)

y_train = train.pop('is_trade')

y_val = val.pop('is_trade')

ins_test = test.pop('instance_id') 

train.pop('instance_id') 

val.pop('instance_id') 

feature_name = val.columns.values.tolist()

del_features = ['realtime','day','context_id'] # ,'search_category_1_cumsum'

for c in del_features:
  if c in feature_name:
    del train[c]
    del val[c]
    del test[c]

feats = val.columns.values.tolist()

print('CSV加载完成')

# V3
lb = LabelEncoder()
feat_set = ['item_id','user_id','shop_id','item_brand_id','item_city_id','category_0','category_1','category_2','property_0','property_1','property_2','predict_category_0','predict_category_1','predict_category_2','predict_category_3']
for feat in feat_set:
    if feat not in del_features:
      print(feat)
      lb.fit_transform((list(train[feat])+list(val[feat])+list(test[feat])))
      train[feat] = lb.transform(list(train[feat]))
      test[feat] = lb.transform(list(test[feat]))
      val[feat] = lb.transform(list(val[feat]))

X_train = train.values
X_val = val.values

model = XGBClassifier( 	learning_rate =0.07,
						 reg_lambda = 80,
						 n_estimators=187,
						 max_depth=7,
						 min_child_weight=2,
						 gamma=0,
						 subsample=0.7,
						 colsample_bytree=0.7,
						 objective= 'binary:logistic',
						 nthread=64,
						 scale_pos_weight=1,
						 seed=1080)

model.fit(X_train, y_train)
# make predictions for test data and evaluate

pro_val = model.predict_proba(X_val)[:,1]

print("Logloss: %.6g" % log_loss(y_val, pro_val))

log_loss(y_val, pro_val)

df = pd.DataFrame({'feature':feats, 'score':importance})

df = df.sort_values(by=['score'])

df.to_csv('model/feature_importance_XGBFS.csv',index=False)

thresholds = list(set(df['score'].values.tolist()))

print(len(thresholds))

#thresholds = list(set(sort(model.feature_importances_)))

Thresh = []
num_feature = []
Logloss = []

for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier(	learning_rate =0.07,
										 reg_lambda = 80,
										 n_estimators=187,
										 max_depth=7,
										 min_child_weight=2,
										 gamma=0,
										 subsample=0.7,
										 colsample_bytree=0.7,
										 objective= 'binary:logistic',
										 nthread=64,
										 scale_pos_weight=1,
										 seed=1080)
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_val = selection.transform(X_val)
	pro_val = selection_model.predict_proba(select_X_val)[:,1]
	Thresh.append(thresh)
	num_feature.append(select_X_train.shape[1])
	Logloss.append(log_loss(y_val, pro_val))
	print("Thresh=%.8g, n=%d, Logloss: %.6g" % (thresh, select_X_train.shape[1], log_loss(y_val, pro_val)))

df = pd.DataFrame({'thresh':Thresh,'num_feature' :num_feature,'Logloss':Logloss})
df = df.sort_values(by=['num_feature'])
df.to_csv('model/XGBFS.csv',index=False)

plt.figure()
plt.plot(df['num_feature'].values.tolist(),df['Logloss'].values.tolist(),'r-')
plt.ylim(0.99*df['Logloss'].min(), 1.01*df['Logloss'].max())
plt.xlabel('num of features')
plt.ylabel('logloss')
plt.gcf().savefig('model/XGBFS_n.png')
