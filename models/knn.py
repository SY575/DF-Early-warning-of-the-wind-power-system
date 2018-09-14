# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:56:39 2018

@author: SY
"""

from sklearn import datasets  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score,recall_score,accuracy_score, roc_auc_score
import numpy as np 
import pandas as pd
from tqdm import tqdm
import os
import math
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)  

data = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

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

_df = data[['ret']].copy()
for f_id in tqdm(list(set(data['multi_label']))):
    temp_val = data.loc[data.multi_label==f_id]
    temp_train = data.loc[data.multi_label!=f_id].reset_index(drop=True)
    temp_ix = temp_val.index.tolist()
    _df['ret'][temp_ix] = temp_val['ret']
    temp_val = temp_val.reset_index(drop=True)
    
    train_x, train_y = temp_train[feat_arr].fillna(0), temp_train['ret'] # data[feat_arr].fillna(0), data['ret']
    val_x, val_y = temp_val[feat_arr].fillna(0), temp_val['ret']
    for col in train_x.columns:
        train_x[col] = train_x[col].apply(lambda x:-1 if math.isinf(x) else x)
    for col in val_x.columns:
        val_x[col] = val_x[col].apply(lambda x:-1 if math.isinf(x) else x)
    
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', 
                               algorithm='auto', leaf_size=100, 
                               p=2, metric='minkowski', 
                               metric_params=None, n_jobs=1)
    knn.fit(train_x, train_y)
    
    y_predict = knn.predict(val_x)
    print(f_id, f1_score(y_predict, val_y))
    
