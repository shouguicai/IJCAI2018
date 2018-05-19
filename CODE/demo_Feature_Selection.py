# -*- coding: utf-8 -*-

from MLFeatureSelection import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np

def loadCSV(path=''):
    return pd.read_csv(path)

def prepareData():
    """prepare you dataset here"""
    '''
    train = loadCSV('data/cutData/train/train_20180425_merge_sel.csv') 
    val = loadCSV('data/cutData/validate/validate_20180425_merge.csv') 
    train.drop_duplicates(subset='instance_id')
    train.pop('instance_id') 
    val.pop('instance_id')
    train = train.fillna(-1)
    val = val.fillna(-1)
    feature_name = val.columns.values.tolist()
    del_features = ['realtime','day','context_id'] 
    feat_import = loadCSV('model/dif_on_feature_mean_4_25.csv')
    del_features += list(feat_import['feature'].values[:43])
    del_features = list(set(del_features))

    for c in del_features:
      if c in feature_name:
        del train[c]
        del val[c]
    
    del train['context_timestamp']

    train['falg'] = 0;
    val['falg'] = 1;
    df = pd.concat([train,val],ignore_index=True)
    print('CSV加载完成')
    return df
    '''
    data = loadCSV('data/cutData/validate/validate_20180425_merge.csv') 
    data.drop_duplicates(subset='instance_id')
    data.pop('instance_id') 
    data = data.fillna(-1)

    feature_name = data.columns.values.tolist()
    del_features = ['realtime','day','context_id'] 
    feat_import = loadCSV('model/dif_on_feature_mean_4_25.csv')
    del_features += list(feat_import['feature'].values[:43])
    del_features = list(set(del_features))

    for c in del_features:
      if c in feature_name:
        del data[c]
    print('CSV加载完成')
    return data


def modelscore(y_test, y_pred):
    """set up the evaluation score"""
    return log_loss(y_test, y_pred)

def validation(X,y,features, clf,lossfunction):
    """set up your validation method"""
    totaltest = 0
    T = (X.context_timestamp >'2018-09-06 23:59:59')
    X_train, X_test = X[~T], X[T]
    X_train, X_test = X_train[features], X_test[features]
    y_train, y_test = y[~T], y[T]
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=50)
    totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
    totaltest /= 1.0
    return totaltest

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}

def main():
    sf = FS.Select(Sequence = False, Random = True, Cross = False) #select the way you want to process searching
    sf.ImportDF(prepareData(),label = 'is_trade')
    sf.ImportLossFunction(modelscore,direction = 'descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['context_timestamp', 'is_trade'])

    features_import = loadCSV('model/dif_on_feature_mean_4_25.csv')
    initial_features = list(features_import['feature'].values)[::-1][:100]
    sf.InitialFeatures(initial_features)
    sf.clf = lgbm.LGBMClassifier(max_depth=7, 
                        n_jobs=60,
                        num_leaves=64,
                        seed=1080,
                        learning_rate=0.05,
                        n_estimators=400,
                        colsample_bytree = 0.8,
                        subsample = 0.8)

    sf.SetLogFile('./model/record_100_7_50.log')
    sf.run(validation)

if __name__ == "__main__":
    main()
