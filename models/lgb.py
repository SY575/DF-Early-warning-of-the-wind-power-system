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
        assert 'multi_label' in test.columns
        _df_test_best = pd.DataFrame(columns=['ret', 'pred', 'file_name'])
        _df_test_worst = pd.DataFrame(columns=['ret', 'pred', 'file_name'])
    _df = data[['ret']].copy()
    _df['pred'] = 0
    for f_id in tqdm(list(set(data['multi_label']))):
        temp_val = data.loc[data.multi_label==f_id]
        temp_train = data.loc[data.multi_label!=f_id].reset_index(drop=True)
        temp_ix = temp_val.index.tolist()
        _df['ret'][temp_ix] = temp_val['ret']
        temp_val = temp_val.reset_index(drop=True)
        '''sl'''
        k = pd.DataFrame()
        def lgb_evalf1(y, pred):
            if len(y) != len(pred):
                return 'logloss',sklearn.metrics.log_loss(y, pred)
            else:
                best_f1 = 0
                # best_thresold=0
                k['res'] = pred
                for i in range(100):
                    threshold = 0.01*i
                    k['pred'] = k['res'].apply(lambda x: 1 if x>threshold else 0)
                    f1 = sklearn.metrics.f1_score(temp_val['ret'].values, k['pred'])
                    if f1 > best_f1:
                        best_f1 = f1
                        # best_thresold=threshold
                return 'f1',best_f1,True
        train_x, train_y = temp_train[feat_arr],temp_train['ret'].values
        train_x_val, train_y_val = temp_val[feat_arr],temp_val['ret'].values
        clf = lgb.LGBMClassifier(boosting_type=lgb_params['boosting_type'], 
                                 num_leaves=lgb_params['num_leaves'], 
                                 reg_alpha=lgb_params['reg_alpha'], 
                                 reg_lambda=lgb_params['reg_lambda'], 
                                 n_estimators=lgb_params['n_estimators'], 
                                 objective=lgb_params['objective'], 
                                 subsample=lgb_params['subsample'], 
                                 colsample_bytree=lgb_params['colsample_bytree'], 
                                 learning_rate=lgb_params['learning_rate'], 
                                 min_child_weight=lgb_params['min_child_weight'],
                                 feature_fraction=1)
        clf.fit(train_x, train_y, eval_set=[(train_x_val, train_y_val)], eval_metric=lgb_evalf1,
                early_stopping_rounds=50, verbose=False)
        _pred = clf.predict_proba(train_x_val)[:,1]
        print(f_id, len(temp_val), roc_auc_score(temp_val.ret, _pred))
        _df['pred'][temp_ix] = _pred
        if test is not None:
            temp_val = test.loc[test.multi_label==f_id]
            _pred = clf.predict_proba(train_x_val)[:,1]
    print('The mean AUC is : %.5f'%roc_auc_score(_df['ret'], _df['pred']))
    if test is not None:
        return _df_test_best, _df_test_worst
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


'''新特征加在这里'''
new_feature = []



feat_arr = list(set(feat_arr).union(set(new_feature)))

data = pd.read_csv('./train.csv')


lgb_params = { 'boosting_type':'gbdt', 'num_leaves':15, 
               'reg_alpha':0., 'reg_lambda':1, 
               'n_estimators':2, 'objective':'binary',
               'subsample':0.7, 'colsample_bytree':0.6, 
               'learning_rate':0.1, 'min_child_weight':1}

_df = fb_cv(data)

_d = _df[['ret', 'pred']].copy()
_pred = pd.DataFrame(_d.pred.tolist(), columns=['score'])
max_arr = [0, 0]
for _T in [_i*0.01 for _i in range(1,100)]:
    _pred['f1_score'] = _pred['score'].apply(lambda x:1 if x > _T else 0)
    _score = f1_score(_d.ret.tolist(), _pred['f1_score'])
    if _score > max_arr[0]:
        max_arr[0] = _score
        max_arr[1] = _T
print('The max F1-score is:', max_arr[0], ', T =', max_arr[1])
