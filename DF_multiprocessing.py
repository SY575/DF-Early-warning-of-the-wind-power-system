# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 00:52:19 2018

@author: SY
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
import pickle
from tqdm import tqdm

def get_feature(args):
    _file_path, mean_arr = args
    _df = pd.read_csv(_file_path)
    _file_name = _file_path.split('/')[-1]
    _dict_return = {}
    _dict_return[_file_name] = []
    
    for col in _df.columns:
        _dict_return[_file_name].append(np.var(_df[col]-mean_arr[col]))
    for col in _df.columns:
        _df[col] = _df[col]/mean_arr[col]
        _dict_return[_file_name].append(np.mean(_df[col]))
        _dict_return[_file_name].append(np.min(_df[col]))
        _dict_return[_file_name].append(np.max(_df[col]))
        _dict_return[_file_name].append(np.ptp(_df[col]))
        _dict_return[_file_name].append(np.median(_df[col]))
        _dict_return[_file_name].append(np.sum(_df[col]))
        
    '''features'''
    _df['_功角'] = _df['变频器电网侧有功功率'] / _df['变频器电网侧无功功率']
    _df['_视在功率'] = (_df['变频器电网侧有功功率'] ** 2 + _df['变频器电网侧无功功率'] ** 2) ** 0.5
    _df['_变频器出入口温差'] = _df['变频器入口温度'] - _df['变频器出口温度']
    _df['_变频器出入口压力'] = _df['变频器入口压力'] - _df['变频器出口压力']
    for col in ['_功角',
                '_视在功率',
                '_变频器出入口温差',
                '_变频器出入口压力']:
        _dict_return[_file_name].append(np.mean(_df[col]))
        _dict_return[_file_name].append(np.min(_df[col]))
        _dict_return[_file_name].append(np.max(_df[col]))
        _dict_return[_file_name].append(np.ptp(_df[col]))
        _dict_return[_file_name].append(np.median(_df[col]))
        _dict_return[_file_name].append(np.sum(_df[col]))
    for col_lst in [['叶片1角度', '叶片2角度', '叶片3角度'],
                    ['变桨电机1电流', '变桨电机2电流','变桨电机3电流'],
                    ['x方向振动值', 'y方向振动值'],
                    ['发电机定子温度1', '发电机定子温度2', '发电机定子温度3', '发电机定子温度4', '发电机定子温度5', '发电机定子温度6'],
                    ['发电机空气温度1', '发电机空气温度2'],
                    ['主轴承温度1', '主轴承温度2'],
                    ['变桨电机1功率估算', '变桨电机2功率估算', '变桨电机3功率估算'],
                    ['叶片1电池箱温度', '叶片2电池箱温度', '叶片3电池箱温度'],
                    ['叶片1变桨电机温度', '叶片2变桨电机温度', '叶片3变桨电机温度'],
                    ['叶片1变频器箱温度', '叶片2变频器箱温度', '叶片3变频器箱温度'],
                    ['叶片1超级电容电压', '叶片2超级电容电压', '叶片3超级电容电压'],
                    ['驱动1晶闸管温度', '驱动2晶闸管温度', '驱动3晶闸管温度'],
                    ['驱动1输出扭矩', '驱动2输出扭矩', '驱动3输出扭矩']]:
        _dict_return[_file_name].append(np.mean([_df[col] for col in col_lst]))
        _dict_return[_file_name].append(np.sum([_df[col] for col in col_lst]))
        _dict_return[_file_name].append(np.var([_df[col] for col in col_lst]))
    return _dict_return


if __name__ == '__main__':
# =============================================================================
#     '''mean_arr'''
#     label = pd.read_csv('./data/train_labels.csv')
#     mean_arr = {}
#     for i, f1 in enumerate(os.listdir('./data/train/')):
#         items = os.listdir('./data/train/'+f1)
#         for item in items:
#             try:
#                 label.loc[label['file_name']==item].ret.values[0]
#             except:
#                 continue
#             if label.loc[label['file_name']==item].ret.values[0] == 0:
#                 _d = pd.read_csv('./data/train/'+f1 + '/' + item)
#                 for col in _d.columns:
#                     if col not in mean_arr:
#                         mean_arr[col] = []
#                     else:
#                         mean_arr[col].append(_d[col].mean())
#     for key in mean_arr.keys():
#         mean_arr[key] = np.mean(mean_arr[key])
#     with open('./mean_arr.plk', 'wb') as f:
#         pickle.dump(mean_arr, f)
# =============================================================================
    with open('./mean_arr.plk', 'rb') as f:
        mean_arr = pickle.load(f)
    '''args'''
    processes = 12
    dict_result = {}
    for f1 in tqdm(os.listdir('./data/train/')):
        with Pool(processes=processes) as pool:
            nargs = os.listdir('./data/train/'+f1)
            nargs = [('./data/train/'+f1 + '/' + _i, mean_arr) for _i in nargs]
            dict_temp = pool.map(get_feature, nargs)
        for item in dict_temp:
            dict_result.update(item)
    data = pd.DataFrame(dict_result)
    data = data.T
    data.columns = [str(i) + '_new' for i in data.columns]
    data['file_name'] = data.index
    data.to_csv('./data/train.csv', index=False)
# =============================================================================
# if __name__ == '__main__':
#     t = time()
#     with open('./mean_arr.plk', 'rb') as f:
#         mean_arr = pickle.load(f)
#     '''args'''
#     processes = 6 # 4
#     dict_result = {}
#     with Pool(processes=processes) as pool:
#         nargs = os.listdir('./data/test/')
#         nargs = [('./data/test/' + _i, mean_arr) for _i in nargs]
#         dict_temp = pool.map(get_feature, nargs)
#     for item in dict_temp:
#         dict_result.update(item)
#     print((time() - t)/60)
#     
#     data = pd.DataFrame(dict_result)
#     data = data.T
#     data.columns = [str(i) + '_new' for i in data.columns]
#     data['file_name'] = data.index
#     data.to_csv('./data/test.csv', index=False)
# =============================================================================
