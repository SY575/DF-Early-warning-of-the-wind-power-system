# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
sys.path.append('./utils/')
from CV import CV
import pandas as pd
import numpy as np

def run(data):
    train = data.loc[data.ret!=-1].reset_index(drop=True)
    test = data.loc[data.ret==-1].reset_index(drop=True)
    feat_arr = ['162', '110', '86', '168', '8', '84', '113', '96', '60', '108',
                '194', '170', '66', '89', '165', '192', '24', '18', '366',
                '258', '354', '360', '11', '276', '120', '158', '270', '246',
                '372', '6', '12', '164', '342', '81', '57', '254', '252',
                '63', '176', '374', '77']
    lgb_params = {'boosting_type':'gbdt', 'num_leaves':150, 
                  'reg_alpha':0., 'reg_lambda':1, 
                  'n_estimators':60, 'objective':'binary',
                  'subsample':0.9, 'colsample_bytree':0.9, 
                  'learning_rate':0.1, 'min_child_weight':5}
    s = CV(_df=train[['ret']+feat_arr], label_name='ret', 
           random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, round_cv=3, n_splits=10)
    pred = s.get_result(test[feat_arr])
    result = test[['file_name']].reset_index(drop=True).copy()
    result['ret'] = pred
    result['ret'].loc[result['ret'] > 0.01] = 1
    result['ret'].loc[result['ret'] <=0.01] = 0
    result = result.rename(columns={'file_name':'id'})
    return result
    
if __name__ == '__main__':
    COL = ['162', '110', '86', '168', '8', '84', '113', '96', '60', '108',
           '194', '170', '66', '89', '165', '192', '24', '18', '366',
           '258', '354', '360', '11', '276', '120', '158', '270', '246',
           '372', '6', '12', '164', '342', '81', '57', '254', '252',
           '63', '176', '374', '77']
    COL += ['file_name']
    data = pd.read_csv('../../data/former/V1/train.csv', usecols=COL)
    test = pd.read_csv('../../data/former/V1/test.csv', usecols=COL)
    label = pd.read_csv('../../data/train_labels.csv')
    data = pd.merge(data, label[['file_name', 'ret']], on='file_name', how='right')
    test['ret'] = -1
    data = data.append(test).reset_index(drop=True)
    result = pd.read_csv('../sub_0808_2.csv').rename(columns={'id':'file_name'})
    baseline = run(data[COL+['ret', 'file_name']])
    
    pred_df = test[['file_name']].copy()
    pred_df = pd.merge(pred_df, result[['file_name', 'ret']], on='file_name', how='left')
    pred_df['pred'] = baseline
    pred_df['temp'] = 0
    T = 0.01
    pred_df['temp'].loc[pred_df['pred']> T] = 1
    pred_df['temp'].loc[pred_df['pred']<=T] = 0
    temp = np.sum(pred_df['temp'] ^ pred_df['ret'])
    print(T, temp)