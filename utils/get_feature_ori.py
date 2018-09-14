# -*- coding: utf-8 -*-

'''本方法仅保留了模型需要的特征'''

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import config
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_feature_ori(args):
    _file_path = args
    _df = pd.read_csv(_file_path)
    _file_name = _file_path.split('/')[-1]
    _dict_return = {}
    _dict_return[_file_name] = []
    
    _COL = ['变频器入口压力_mean', '变频器电网侧电压_max', '风向绝对值_max', '变频器出口压力_mean',
            '轮毂角度_max', '风向绝对值_mean', '变频器电网侧电压_median', '无功功率控制状态_mean', 'x方向振动值_mean',
            '变频器电网侧电压_mean', '测风塔环境温度_max', '变频器出口压力_max', 'y方向振动值_mean',
            '风向绝对值_median', '变频器入口压力_var', '测风塔环境温度_mean', '叶片3角度_mean',
            '叶片2角度_mean', '叶片2变桨电机温度_mean', '轮毂温度_mean', '叶片3电池箱温度_mean',
            '叶片1变桨电机温度_mean', '轮毂角度_median', '机舱控制柜温度_mean', '变频器电网侧无功功率_mean', '变频器出口温度_max',
            '机舱温度_mean', '主轴承温度1_mean', '叶片3变桨电机温度_mean', '轮毂角度_mean',
            '叶片1角度_mean', '变频器入口压力_max', '叶片1电池箱温度_mean', '机舱气象站风速_var',
            '5秒偏航对风平均值_var', '主轴承温度2_max', '主轴承温度2_mean', 'x方向振动值_var',
            '发电机功率限幅值_max', '叶片3变桨电机温度_max', '液压制动压力_median']
    for col in _COL:
        if 'mean' in col:
            _dict_return[_file_name].append(np.mean(_df[col.split('_')[0]]))
        elif 'min' in col:
            _dict_return[_file_name].append(np.min(_df[col.split('_')[0]]))
        elif 'max' in col:
            _dict_return[_file_name].append(np.max(_df[col.split('_')[0]]))
        elif 'var' in col:
            _dict_return[_file_name].append(np.var(_df[col.split('_')[0]]))
        elif 'ptp' in col:
            _dict_return[_file_name].append(np.ptp(_df[col.split('_')[0]]))
        elif 'median' in col:
            _dict_return[_file_name].append(np.median(_df[col.split('_')[0]]))
    
    
    return _dict_return


def run():
    if 'data_ori.csv' in os.listdir('./'):
        return pd.read_csv('./data_ori.csv')
    elif 'data_ori.csv' in os.listdir('../'):
        return pd.read_csv('../data_ori.csv')
    COL = ['变频器入口压力_mean', '变频器电网侧电压_max', '风向绝对值_max', '变频器出口压力_mean',
           '轮毂角度_max', '风向绝对值_mean', '变频器电网侧电压_median', '无功功率控制状态_mean', 'x方向振动值_mean',
           '变频器电网侧电压_mean', '测风塔环境温度_max', '变频器出口压力_max', 'y方向振动值_mean',
           '风向绝对值_median', '变频器入口压力_var', '测风塔环境温度_mean', '叶片3角度_mean',
           '叶片2角度_mean', '叶片2变桨电机温度_mean', '轮毂温度_mean', '叶片3电池箱温度_mean',
           '叶片1变桨电机温度_mean', '轮毂角度_median', '机舱控制柜温度_mean', '变频器电网侧无功功率_mean', '变频器出口温度_max',
           '机舱温度_mean', '主轴承温度1_mean', '叶片3变桨电机温度_mean', '轮毂角度_mean',
           '叶片1角度_mean', '变频器入口压力_max', '叶片1电池箱温度_mean', '机舱气象站风速_var',
           '5秒偏航对风平均值_var', '主轴承温度2_max', '主轴承温度2_mean', 'x方向振动值_var',
           '发电机功率限幅值_max', '叶片3变桨电机温度_max', '液压制动压力_median']
        
    label = pd.read_csv(config.LABEL_PATH)
        
    # train
    dict_result = {}
    for f1 in tqdm(os.listdir(config.TRAIN_PATH)):
        dict_temp = []
        nargs = os.listdir(config.TRAIN_PATH + f1)
        nargs = [(config.TRAIN_PATH + f1 + '/' + _i) for _i in nargs]
        for item in nargs:
            dict_temp.append(get_feature_ori(item))
        for item in dict_temp:
            dict_result.update(item)
# =============================================================================
#         break
# =============================================================================
    train = pd.DataFrame(dict_result)
    train = train.T
    train.columns = COL
    train['file_name'] = train.index
    train = train.reset_index(drop=True)
    
    # test
    dict_result = {}
    dict_temp = []
    nargs = os.listdir(config.TEST_PATH)
    nargs = [(config.TEST_PATH + _i) for _i in nargs]
    for item in tqdm(nargs):
        dict_temp.append(get_feature_ori(item))
# =============================================================================
#         break
# =============================================================================
    for item in dict_temp:
        dict_result.update(item)
    test = pd.DataFrame(dict_result)
    test = test.T
    test.columns = COL
    test['file_name'] = test.index
    test = test.reset_index(drop=True)
    
    train = pd.merge(train, label[['file_name', 'ret']], on='file_name', how='right')
    train = train.reset_index(drop=True)
    test['ret'] = -1
    train = train.append(test).reset_index(drop=True)
    return train

if __name__ == '__main__':
    data = run()