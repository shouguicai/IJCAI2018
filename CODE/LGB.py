# -*- coding: utf-8 -*-

import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

def loadCSV(path=''):
    return pd.read_csv(path)

online = True # 这里用来标记是 线下验证 还是 在线提交
'''
train = loadCSV('data/cutData/train/train_20180425_merge_sel.csv') 

val = loadCSV('data/cutData/validate/validate_20180425_merge.csv') 
'''
data = loadCSV('data/cutData/validate/validate_20180425_merge.csv') 

T = (data.context_timestamp >'2018-09-06 23:59:59')

del data['context_timestamp']

train, val = data[~T], data[T]

del data

test = loadCSV('data/cutData/test/test_20180425_merge.csv')

train = train.fillna(-1)

val = val.fillna(-1)

test = test.fillna(-1)

train.drop_duplicates(subset='instance_id')

y_train = train.pop('is_trade')

y_val = val.pop('is_trade')

ins_test = test.pop('instance_id') 

train.pop('instance_id') 

val.pop('instance_id') 

feature_name = val.columns.values.tolist()

del_features = ['realtime','day','context_id'] # ,'search_category_1_cumsum'

feat_import = loadCSV('model/dif_on_feature_mean_4_25.csv')

del_features += list(feat_import['feature'].values[:43])

del_features = list(set(del_features))

for c in del_features:
  if c in feature_name:
    del train[c]
    del val[c]
    del test[c]

print('CSV加载完成')

feature_name = val.columns.values.tolist()

X_train = train[feature_name]

X_val = val[feature_name]

X_test = test[feature_name]

del train

del test

del val

cat_feature = ['user_gender_id']

'''
clf = lgb.LGBMClassifier(max_depth=7, 
                        n_jobs=60,
                        #objective='binary',
                        num_leaves=64,
                        seed=1080,
                        learning_rate=0.01,
                        n_estimators=2000,
                        colsample_bytree = 0.8,
                        subsample = 0.8)

clf.fit(X_train, y_train,feature_name=feature_name,eval_set=[(X_train, y_train),(X_val, y_val)],
categorical_feature=cat_feature,early_stopping_rounds=50)
'''
#best_iter = clf.best_iteration_

best_iter = 2000

if online == True:

    # 0.1 73, 0.01 725
    # 0.05 152
    X_train = X_val

    y_train = y_val
    #X_train = pd.concat([X_train,X_val],ignore_index=True)

    #y_train = pd.concat([y_train,y_val],ignore_index=True)

    del X_val

    del y_val

    clf = lgb.LGBMClassifier(max_depth=7, 
                            n_jobs=60,
                            #objective='binary',
                            num_leaves=64,
                            learning_rate=0.01,
                            n_estimators=best_iter,
                            colsample_bytree = 0.8,
                            subsample = 0.8)

    clf.fit(X_train, y_train,eval_set=[(X_train, y_train)],
            categorical_feature=cat_feature)

    prob = clf.predict_proba(X_test)[:, 1] 

    output = pd.DataFrame({'instance_id':ins_test,'predicted_score':prob})

    initial_ins = pd.read_csv('result/initial_instance_id.csv') 

    output = pd.merge(initial_ins,output,how='left',on='instance_id')

    output.to_csv('result/submission_lgb_5_8_del_mean_dif_43_set_7.txt', index=False, sep=' ', columns={'instance_id', 'predicted_score'})

    print('结果预测完成')