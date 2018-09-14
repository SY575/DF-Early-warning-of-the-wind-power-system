# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../')
from CV import CV
import config
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def run(data, result_temp):
    test = data.loc[data.ret==-1].reset_index(drop=True)
    data = data.loc[data.ret!=-1].reset_index(drop=True)
    file_name_dict = {}
    for f1 in os.listdir(config.TRAIN_PATH):
        for f2 in os.listdir(config.TRAIN_PATH+f1):
            file_name_dict[f2] = int(f1)
    data['multi_label'] = data.file_name.apply(lambda x:file_name_dict[x])
    data = data.loc[data['multi_label']!=14].reset_index(drop=True)
    
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=10, 
                             learning_rate=0.1, n_estimators=100, 
                             subsample_for_bin=200000, objective='multiclass', 
                             min_child_weight=1, min_child_samples=20, 
                             subsample=0.7, subsample_freq=0, 
                             colsample_bytree=0.7, 
                             reg_alpha=0.0, reg_lambda=0.0, 
                             random_state=3)
    
    train_x, val_x, train_y, val_y = train_test_split(data.drop(['file_name', 'ret', 'multi_label'], axis=1), 
                                                      data['multi_label'], 
                                                      random_state=3, 
                                                      test_size=0.3)
    clf.fit(train_x, train_y, verbose=False,early_stopping_rounds=100, eval_metric='logloss', eval_set=[(val_x, val_y)])
    
    pred_val = clf.predict(val_x)
    result_val = pd.DataFrame(index=list(range(len(val_x))))
    result_val['label'] = val_y.tolist()
    result_val['pred'] = pred_val
    
    pred_test = clf.predict(test[train_x.columns.tolist()])
    result_test = pd.DataFrame(index=list(range(len(test))))
    result_test['pred'] = pred_test
    test['multi_label'] = pred_test
    pred = clf.predict_proba(test[train_x.columns.tolist()])
    temp = []
    for i in range(len(pred)):
        temp.append(np.max(pred[i]))
    test['prob'] = temp
    
    
    '''单独训练'''
    print('training...')
    result_dict = {}
    result_prob_dict = {}
    c = Counter(data.multi_label)
    for class_ in tqdm(list(c.keys())):
        lgb_params = { 'boosting_type':'gbdt', 'num_leaves':8, 
                   'reg_alpha':0., 'reg_lambda':1, 
                   'n_estimators':30, 'objective':'binary',
                   'subsample':0.7, 'colsample_bytree':0.6, 
                   'learning_rate':0.1, 'min_child_weight':1}
        s = CV(_df=data.loc[data.multi_label==class_].drop(['file_name', 'multi_label'], axis=1).reset_index(drop=True), 
                     label_name='ret')
        s.CV(is_print=False, lgb_params=lgb_params, round_cv=3, n_splits=8) # , eval_metrics=f1_score
        test_temp = test.loc[test.multi_label==class_].reset_index(drop=True)
        pred_temp = s.get_result(test_temp.drop(['file_name', 'multi_label','prob', 'ret'], axis=1))
        for i in range(len(test_temp)):
            result_dict[test_temp['file_name'][i]] = pred_temp[i]
            result_prob_dict[test_temp['file_name'][i]] = test_temp['prob'][i]
    
            
    df = pd.DataFrame(index=range(len(result_dict)))
    df['id'] = result_dict.keys()
    df['ret'] = result_dict.values()
    df['prob'] = result_prob_dict.values()
    
    df['multi_score'] = 2*(1-df.ret)**2*df.prob/((1-df.ret)**2+df.prob)

    dict_ = {}
    tp_df = df.loc[np.logical_and(df.prob>0.999, df.ret<0.1)].copy()
    tp_df = tp_df.reset_index(drop=True)
    for i in range(len(tp_df)):
        dict_[tp_df['id'][i]] = 0
    print(len(dict_))
    result = result_temp.copy()
    result['pred_2'] = result['id'].apply(lambda x:0 if x in dict_ else 1)
    result['pred_2'] = result['pred_2'] * result['ret']
    r = result[['id', 'pred_2']].copy()
    r.columns = ['id', 'ret']
    r['ret'] = r['ret'].astype(int)
    return r

if __name__ == '__main__':
    train = pd.read_csv('../../data/train.csv')
    label = pd.read_csv('../../data/train_labels.csv')
    train = pd.merge(train, label[['file_name', 'ret']], on='file_name', how='right')
    test = pd.read_csv('../../data/test.csv')
    test['ret'] = -1
    label = run(train.append(test).reset_index(drop=True), pd.read_csv('../temp/sub_0808_2.csv'))