# -*- coding: utf-8 -*-

'''本方法仅保留了模型需要的特征'''


feat_arr = ['变频器电网侧电压_max', '变频器入口压力_mean', '无功功率控制状态_sum', '变频器出口压力_mean',
            '_变频器出入口压力_min', '轮毂角度_median', '变频器出口压力_max', '叶片3角度_median',
            '发电机功率限幅值_mean', '轮毂角度_max', '_变频器出入口压力_max', 'x方向振动值_y方向振动值_mean',
            '变频器电网侧电压_mean', '变频器电网侧电压_median', 'x方向振动值_mean', '风向绝对值_max',
            '叶片1角度_max', '无功功率控制状态_mean', '变频器入口压力_sum', '_功角_median',
            '叶片1角度_median', 'y方向振动值_sum', '_变频器出入口压力_median', '超速传感器转速检测值_mean',
            '叶片1变频器箱温度_叶片2变频器箱温度_叶片3变频器箱温度_mean', '叶片2角度_median',
            '叶片3变桨电机温度_mean', '叶片1超级电容电压_median', 'y方向振动值_mean', 'y方向振动值_max',
            '风向绝对值_mean', '叶片3变桨电机温度_sum', '_变频器出入口温差_ptp', '_变频器出入口压力_mean',
            '变频器入口压力_max', '叶片1变桨电机温度_叶片2变桨电机温度_叶片3变桨电机温度_mean',
            'y方向振动值_median', '风向绝对值_median', '变频器电网侧电流_median', '机舱控制柜温度_median']

feat_arr_01 = ['变频器入口压力_mean', '变频器电网侧电压_max', '风向绝对值_max', '变频器出口压力_mean',
               '轮毂角度_max', '风向绝对值_mean', '变频器电网侧电压_median', '无功功率控制状态_mean', 'x方向振动值_mean',
               '变频器电网侧电压_mean', '测风塔环境温度_max', '变频器出口压力_max', 'y方向振动值_mean',
               '风向绝对值_median', '变频器入口压力_var', '测风塔环境温度_mean', '叶片3角度_mean',
               '叶片2角度_mean', '叶片2变桨电机温度_mean', '轮毂温度_mean', '叶片3电池箱温度_mean',
               '叶片1变桨电机温度_mean', '轮毂角度_median', '机舱控制柜温度_mean', '变频器电网侧无功功率_mean', '变频器出口温度_max',
               '机舱温度_mean', '主轴承温度1_mean', '叶片3变桨电机温度_mean', '轮毂角度_mean',
               '叶片1角度_mean', '变频器入口压力_max', '叶片1电池箱温度_mean', '机舱气象站风速_var',
               '5秒偏航对风平均值_var', '主轴承温度2_max', '主轴承温度2_mean', 'x方向振动值_var',
               '发电机功率限幅值_max', '叶片3变桨电机温度_max', '液压制动压力_median']
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import config
import pandas as pd
import numpy as np
import os
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
    '''构建'''
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

def run():
    if 'data.csv' in os.listdir('./'):
        return pd.read_csv('./data.csv')
    elif 'data.csv' in os.listdir('../'):
        return pd.read_csv('../data.csv')
        
    label = pd.read_csv(config.LABEL_PATH)
# =============================================================================
#     mean_arr = {}
#     for f1 in tqdm(os.listdir(config.TRAIN_PATH)):
#         items = os.listdir(config.TRAIN_PATH + f1)
#         for item in items:
#             try:
#                 label.loc[label['file_name']==item].ret.values[0]
#             except:
#                 continue
#             if label.loc[label['file_name']==item].ret.values[0] == 0:
#                 _d = pd.read_csv(config.TRAIN_PATH + f1 + '/' + item)
#                 for col in _d.columns:
#                     if col not in mean_arr:
#                         mean_arr[col] = []
#                     else:
#                         mean_arr[col].append(_d[col].mean())
# # =============================================================================
# #         break
# # =============================================================================
#     for key in mean_arr.keys():
#         mean_arr[key] = np.mean(mean_arr[key])
# =============================================================================
    import pickle
    with open('../../../former/mean_arr.plk', 'rb') as f:
        mean_arr = pickle.load(f)
    # train
    dict_result = {}
    for f1 in tqdm(os.listdir(config.TRAIN_PATH)):
        dict_temp = []
        nargs = os.listdir(config.TRAIN_PATH + f1)
        nargs = [(config.TRAIN_PATH + f1 + '/' + _i, mean_arr) for _i in nargs]
        for item in nargs:
            dict_temp.append(get_feature(item))
        for item in dict_temp:
            dict_result.update(item)
        break
    train = pd.DataFrame(dict_result)
    train = train.T
    train.columns = [str(i)+'_new' for i in range(588)]
    train['file_name'] = train.index
    train = train.reset_index(drop=True)
    
    # test
    dict_result = {}
    dict_temp = []
    nargs = os.listdir(config.TEST_PATH)
    nargs = [(config.TEST_PATH + _i, mean_arr) for _i in nargs]
    for item in tqdm(nargs):
        dict_temp.append(get_feature(item))
        break
    for item in dict_temp:
        dict_result.update(item)
    test = pd.DataFrame(dict_result)
    test = test.T
    test.columns = [str(i)+'_new' for i in range(588)]
    test['file_name'] = test.index
    test = test.reset_index(drop=True)
    
    train = pd.merge(train, label[['file_name', 'ret']], on='file_name', how='right')
    train = train.reset_index(drop=True)
    test['ret'] = -1
    train = train.append(test).reset_index(drop=True)
    return train

if __name__ == '__main__':
    data = run()