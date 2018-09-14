# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:01:24 2018

@author: SY
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
from sklearn.metrics import f1_score,recall_score,accuracy_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

def fb_cv(data, test=None):
    assert 'multi_label' in data.columns
    assert feat_arr is not None
    if test is not None:
        _df_test = test[['file_name']].copy()
        _df_test['pred'] = 0
        _df_test_pred_lst = []
    _df = data[['ret']].copy()
    _df['pred'] = 0
    for f_id in tqdm(list(set(data['multi_label']))):
        # split
        temp_val = data.loc[data.multi_label==f_id]
        temp_train = data.loc[data.multi_label!=f_id].reset_index(drop=True)
        temp_ix = temp_val.index.tolist()
        _df['ret'][temp_ix] = temp_val['ret']
        temp_val = temp_val.reset_index(drop=True)
        # train
        k = pd.DataFrame()
        def evalf1(preds,dtrain):
            labels=dtrain.get_label()
            if len(temp_val)!=len(preds):
                return 'logloss',sklearn.metrics.log_loss(labels,preds)
            else:
                best_f1=0
                # best_thresold=0
                k['res']=preds
                for i in range(100):
                    threshold=0.01*i
                    k['pred']=k['res'].apply(lambda x: 1 if x>threshold else 0)
                    f1=sklearn.metrics.f1_score(temp_val['ret'].values, k['pred'])
                    if f1>best_f1:
                        best_f1=f1
                        # best_thresold=threshold
                return 'f1',best_f1
        xgb_train = xgb.DMatrix(temp_train[feat_arr],temp_train['ret'].values)
        xgb_valid = xgb.DMatrix(temp_val[feat_arr],temp_val['ret'].values)
        watch_list = [(xgb_train,'dtrain'), 
                      (xgb_valid,'dvalid')]
        xgb_model = xgb.train(params,xgb_train, 2000,
                              watch_list,feval=evalf1,
                              early_stopping_rounds=50,
                              verbose_eval=5,maximize=True)
        # predict
        _pred = xgb_model.predict(xgb.DMatrix(temp_val[feat_arr]))
        print('\n', f_id, len(temp_val), roc_auc_score(temp_val.ret, _pred))
        _df['pred'][temp_ix] = _pred
        if test is not None:
            _pred = xgb_model.predict(xgb.DMatrix(test[feat_arr]))
            _df_test_pred_lst.append(_pred)
            
    print('The mean AUC is : %.5f'%roc_auc_score(_df['ret'], _df['pred']))
    if test is not None:
        pred_lst = []
        for i in range(len(_df_test_pred_lst[0])):
            pred_lst.append(np.mean([_df_test_pred_lst[j][i] for j in range(len(_df_test_pred_lst))]))
        _df_test['pred'] = pred_lst
        return _df, _df_test
    return _df

feat_arr = [ '变频器电网侧电压_max', '变频器入口压力_mean', '无功功率控制状态_sum',
             '变频器出口压力_mean', '_变频器出入口压力_min', '轮毂角度_median', '变频器出口压力_max',
             '叶片3角度_median', '发电机功率限幅值_mean', '轮毂角度_max', '_变频器出入口压力_max',
             '__变_频_器_出_入_口_压_力_mean', '变频器电网侧电压_mean', '变频器电网侧电压_median',
             'x方向振动值_mean', '风向绝对值_max', '叶片1角度_max', '无功功率控制状态_mean',
             '变频器入口压力_sum', '_功角_median', '叶片1角度_median', 'y方向振动值_sum',
             '_变频器出入口压力_median', '超速传感器转速检测值_mean', '__变_频_器_出_入_口_压_力_mean.1',
             '叶片2角度_median', '叶片3变桨电机温度_mean', '叶片1超级电容电压_median', 'y方向振动值_mean',
             'y方向振动值_max', '风向绝对值_mean', '叶片3变桨电机温度_sum', '_变频器出入口温差_ptp',
             '_变频器出入口压力_mean', '变频器入口压力_max', '__变_频_器_出_入_口_压_力_mean.2',
             'y方向振动值_median', '风向绝对值_median', '变频器电网侧电流_median', '机舱控制柜温度_median']


data = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',#'rank:pairwise'
    'eta': 0.1, # style 2 0.033
    'seed' : 0,
    'max_depth': 4,# style 2 5
    'subsample': 0.8, # 0.95
    'colsample_bytree': 0.8,
#     'colsample_bylevel' : 0.5,
    'min_child_weight': 1,
    'eval_metric': ['logloss'],
    'nthread' : 8,
    'gamma': 1,
    'lambda' : 1,
    'alpha': 1
}


_df, _df_test = pre_train_score = fb_cv(data, test)
