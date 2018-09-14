# -*- coding: utf-8 -*-

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from CV import CV
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
import config

def run(data, result_best):
    feat_arr = ['185_new', '237_new', '176_new', '243_new', '544_new', '85_new',
                '245_new', '103_new', '249_new', '83_new', '545_new', '555_new', '183_new',
                '187_new', '135_new', '161_new', '89_new', '171_new', '242_new', '529_new',
                '91_new', '146_new', '547_new', '123_new', '576_new', '97_new', '447_new',
                '475_new', '141_new', '143_new', '159_new', '452_new', '540_new', '543_new',
                '239_new', '573_new', '145_new', '163_new', '181_new', '355_new']
    # 名字转换
    temp_1 = os.listdir(config.TRAIN_PATH)[0]
    d = pd.read_csv(config.TRAIN_PATH+'/'+temp_1+'/'+os.listdir(config.TRAIN_PATH+'/'+temp_1)[0])
    name_lst = []
    for col in d.columns:
        name_lst.append(col+'_var')
    for col in d.columns.tolist() + ['_功角','_视在功率','_变频器出入口温差','_变频器出入口压力']:
        name_lst.append(col+'_mean')
        name_lst.append(col+'_min')
        name_lst.append(col+'_max')
        name_lst.append(col+'_ptp')
        name_lst.append(col+'_median')
        name_lst.append(col+'_sum')
    for col in [['叶片1角度', '叶片2角度', '叶片3角度'],
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
        name_lst.append('_'.join(col)+'_mean')
        name_lst.append('_'.join(col)+'_sum')
        name_lst.append('_'.join(col)+'_var')
    dict_name = {}
    col_lst = data.columns.tolist()[:-1]
    for i in range(len(name_lst)):
        dict_name[col_lst[i]] = name_lst[i]
    
    data = data[feat_arr+[str(name_lst.index('液压制动压力_max'))+'_new', 'ret', 'file_name']]
    data.columns = [dict_name[i] for i in feat_arr+[str(name_lst.index('液压制动压力_max'))+'_new']] + ['ret', 'file_name']
    
    test = data.loc[data.ret==-1].reset_index(drop=True)
    data = data.loc[data.ret!=-1].reset_index(drop=True)

    file_name_dict = {}
    for f1 in os.listdir(config.TRAIN_PATH):
        for f2 in os.listdir(config.TRAIN_PATH+f1):
            file_name_dict[f2] = int(f1)
    
    data['multi_label'] = data.file_name.apply(lambda x:file_name_dict[x])
    data_14 = data.loc[data['multi_label']==14].reset_index(drop=True)
    data = data.loc[data['multi_label']!=14].reset_index(drop=True)
    
    
    lgb_params = { 'boosting_type':'gbdt', 'num_leaves':8, 
                   'reg_alpha':0., 'reg_lambda':1, 
                   'n_estimators':50, 'objective':'binary',
                   'subsample':0.7, 'colsample_bytree':0.6, 
                   'learning_rate':0.1, 'min_child_weight':1}
    feat_arr = [dict_name[i] for i in feat_arr]
# =============================================================================
#     '''6751 - 6755  test_02.csv 0816'''
# =============================================================================
    temp_test = test.loc[np.logical_and(np.logical_and(test['液压制动压力_max']<1.32,
                                                       test['液压制动压力_max']>1), 
                                        test['x方向振动值_mean']<-1.5)]
    temp_val = temp_test # test
    temp_train = data
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.2:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.2:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    '''6755 - 6772  submission_3.csv 0816'''
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<3.4,
                                                   test['x方向振动值_mean']>1.2),
                                    np.logical_and(test['y方向振动值_mean']<3,
                                                   test['y方向振动值_mean']>2))]
    
    temp_val = temp_test # test
    temp_train = data
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    '''6772 - 6773  test.csv 0816'''
    temp_test = test.loc[np.logical_and(np.logical_and(test['液压制动压力_max']>1.32,
                                                        test['液压制动压力_max']>1),
            np.logical_and(np.logical_and(test['x方向振动值_mean']<-0.3,
                                                       test['x方向振动值_mean']<22.05),
                                        np.logical_and(test['y方向振动值_mean']<.8,
                                                       test['y方向振动值_mean']>0)))]
    
    temp_val = temp_test # test
    temp_train = data
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    '''6773 - 6816'''
    test['temp'] = test['x方向振动值_mean'] + 1.2 - test['y方向振动值_mean']
    temp_test = test.loc[np.logical_and(np.logical_and(np.logical_and(test['x方向振动值_mean']<0.34,# 0.34
                                                                                     test['x方向振动值_mean']>-0.25),
                                                    np.logical_and(test['y方向振动值_mean']>0,
                                                       test['y方向振动值_mean']<1.8)),
                        test['temp']<0)]
    temp_val = temp_test # test
    temp_train = data
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/678.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.18:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    temp_val = temp_test # test
    temp_train = data
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] > 0.18:
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] > 0.18:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    '''6822 - 6834'''
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<2,
                                                       test['x方向振动值_mean']>1.25),
                                        np.logical_and(test['y方向振动值_mean']<3.2,
                                                       test['y方向振动值_mean']>2))]
    
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.34: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6822.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6816.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.34: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_best_2 = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35 and temp_result.pred[i] > 0.35:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    result_best_2.ret[ix_lst] = 1
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6816.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_best_2 = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35 and temp_result.pred[i] > 0.35:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    result_best_2.ret[ix_lst] = 1
    result_best_2 = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.35 and temp_result.pred[i] > 0.34:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    result_best_2.ret[ix_lst] = 1
    result_best = result_best_2.copy()
    test['temp'] = test['x方向振动值_mean'] + 0.75 - test['y方向振动值_mean']
    temp_test = test.loc[np.logical_and(np.logical_and(np.logical_and(test['y方向振动值_mean']<1.25, # 1.25 , 1
                                                       test['y方向振动值_mean']>0.75), # 0.75, 1
                                        np.logical_and(test['x方向振动值_mean']>0.25,
                                                       test['x方向振动值_mean']<0.6)),test['temp']>0)]
    
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6816_new.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.45: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6822.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.45: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6822.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6816_new.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4: # 0.18 0.27
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.4:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<1.7,
                                                       test['x方向振动值_mean']>1.36),
                                        np.logical_and(test['y方向振动值_mean']>1.4,
                                                       test['y方向振动值_mean']<1.8))]
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/6822_new.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.27: # 0.18 0.27   0.4  
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_best.ret.sum()
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.27:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
# =============================================================================
#     result_sub.to_csv('../V9_final/result/0820_1.csv', index=False)
#     result_best = pd.read_csv('../V9_final/result/0820_1.csv')
# =============================================================================
    result_best = result_sub.copy()
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<1,
                                                       test['x方向振动值_mean']>0.8),
                                        np.logical_and(test['y方向振动值_mean']>0.9,
                                                       test['y方向振动值_mean']<1.1))]
    
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/0820_1.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.39: # 0.18 0.27   0.4 0.27 
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.39: # 0.18 0.27   0.4 0.27 
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.39:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    
    test['temp'] = test['x方向振动值_mean'] + 0.2 - test['y方向振动值_mean']
    temp_test = test.loc[np.logical_and(np.logical_and(
                                    test['液压制动压力_max']>1,
                                    np.logical_and(np.logical_and(test['x方向振动值_mean']<0.4,
                                                                                     test['x方向振动值_mean']>-0.3),
                                                    np.logical_and(test['y方向振动值_mean']>-0.3,
                                                       test['y方向振动值_mean']<0.14))),
                        test['temp']>0)]
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    # result_best = pd.read_csv('../V9_final/result/0820_2.csv')
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.3: # 0.18 0.27   0.4  0.27  0.39
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_best.ret.sum()
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.3:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<3,
                                                       test['x方向振动值_mean']>1.8),
                                        np.logical_and(test['y方向振动值_mean']>1,
                                                       test['y方向振动值_mean']<2.1))]
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.38: # 0.18 0.27   0.4  0.27  0.39  0.3
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.37: # 0.18 0.27   0.4  0.27  0.39  0.3
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.37:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<-0.8, # -0.55
                                                       test['x方向振动值_mean']>-2),
                                        np.logical_and(test['y方向振动值_mean']>-0.9,
                                                       test['y方向振动值_mean']<-0.2))]
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.25: # 0.18 0.27   0.4  0.27  0.39  0.3  0.37
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.25:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_best = result_sub.copy()
    temp_test = test.loc[np.logical_and(np.logical_and(test['x方向振动值_mean']<1,
                                                       test['x方向振动值_mean']>0),
                                        np.logical_and(test['y方向振动值_mean']>1.9,
                                                       test['y方向振动值_mean']<2.5))]
    
    temp_val = temp_test # test
    temp_train = data# .loc[:len(data)-1454-2496-1] # -2496, 1454
    s = CV(_df=temp_train[['ret']+feat_arr], label_name='ret', 
                 random_state=3, is_val=False)
    s.CV(is_print=False, lgb_params=lgb_params, n_splits=5, round_cv=1)
    pred = s.get_result(temp_val[feat_arr])
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.2:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_sub.ret.sum()
    temp_result = temp_val[['file_name']].reset_index(drop=True).copy()
    temp_result['pred'] = pred
    
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = result_best.ret[i]
    
    temp_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.2: # 0.18 0.27   0.4  0.27  0.39  0.3  0.37  0.25
            temp_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub = result_best.copy()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] < 0.2:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 0
    result_sub.ret.sum()
    dict_result_best = {}
    for i in range(len(result_best)):
        dict_result_best[result_best.id[i]] = i
    
    ix_lst = []
    for i in range(len(temp_result)):
        if temp_result.pred[i] >0.8:
            ix_lst.append(dict_result_best[temp_result.file_name[i]])
    
    result_sub.ret[ix_lst] = 1
    result_best = result_sub.copy()
    return result_best


if __name__ == '__main__':
# =============================================================================
#     train = pd.read_csv('../../../data/train.csv')
#     label = pd.read_csv('../../../data/train_labels.csv')
#     train = pd.merge(train, label[['file_name', 'ret']], on='file_name', how='right')
#     test = pd.read_csv('../../../data/test.csv')
#     test['ret'] = -1
#     temp = run(train.append(test).reset_index(drop=True), 
#                pd.read_csv('../../temp/sub_02.csv'))
# =============================================================================
    temp = run(pd.read_csv('../../../data.csv'), 
               pd.read_csv('../../temp/sub_02.csv'))
