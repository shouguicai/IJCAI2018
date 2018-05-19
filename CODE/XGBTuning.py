# -*- coding: utf-8 -*-

# XGB 调参

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  

import matplotlib as mpl
mpl.use('Agg') # 没有GUI时使用matplotlib绘图,写在import pylab前
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import log_loss
rcParams['figure.figsize'] = 12, 4
target = 'is_trade'

def loadCSV(path=''):
    return pd.read_csv(path)

def modelfit(alg, X_train,y_train,useTrainCV=False, cv_folds=6, early_stopping_rounds=100):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgb_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds,show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train,y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
        
    #Print model report:
    print "Model Report"
    print "Log Loss: %.4g" % log_loss(y_train, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob)
    
    '''               
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.gcf().savefig('model/feature_importance_xgb_T.png')
    print('Feature importance 已保存')
    '''

train = loadCSV('data/cutData/train/train_20180301_merge.csv') 

print('CSV加载完成')

y_train = train.pop('is_trade')

train.pop('instance_id') 


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
'''
# V2
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

X_train = train.drop(['item_id','user_id','context_id','shop_id',
                      'category_0','category_1','category_2','property_0',
                      'property_1','property_2','predict_category_0',
                      'predict_category_1','predict_category_2'],axis=1)

#feature_name = X_train.columns.values.tolist()

xgb1 = XGBClassifier(
 learning_rate =0.07,
 reg_lambda = 80,
 n_estimators=300,
 max_depth=7,
 min_child_weight=1,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=64,
 scale_pos_weight=1,
 seed=1080)

# 运行
print('开始训练')
modelfit(xgb1, X_train,y_train)
