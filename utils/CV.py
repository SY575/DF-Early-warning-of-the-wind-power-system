# -*- coding: utf-8 -*-
"""
Created on Mon May 21 23:19:34 2018

@author: SY
"""
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class CV(object):
    def __init__(self, _df, _val=None, 
                 label_name='label', pred_col_name='score',
                 random_state=575, is_val=True, val_size=0.3):
        self.is_val = is_val
        if pred_col_name in _df.columns.tolist():
            self.score = _df[pred_col_name].values.tolist()
        else:
            self.score = None
        if label_name in _df.columns.tolist():
            self.label = _df[label_name].values.tolist()
        else:
            print("Attention! The dataset don't have the %s"%(label_name))
            self.label = None
        if _val is not None:
            self.df = _df
            self.val = _val
        else:
            self.df, self.val, _, _ = train_test_split(_df, 
                                                       _df[label_name], 
                                                       test_size=val_size,
                                                       random_state=random_state)
            if self.is_val:
                self.df = self.df.reset_index(drop=True)
                self.val = self.val.reset_index(drop=True)
            else:
                self.df = _df
                self.val = self.val.reset_index(drop=True)
        # others
        self.random_state = random_state
        self.label_name = label_name
        self.pred_col_name = pred_col_name
        self.AUC = 0
        self.find_best = None
        self.clf_arr = []
        self.MS_arr = None

    
    def CV(self, _df=None, _df_val=None, round_cv=3, 
           n_splits=5, is_print=False, replace=False,
           eval_metrics=roc_auc_score, 
           use_lgb=True, lgb_params=None):
        '''init'''
        self.is_print = is_print
        self.eval_metrics = eval_metrics
        self.use_lgb = use_lgb
        self.lgb_params = lgb_params
        if _df is None:
            train_df = self.df
        else:
            if len(_df)!=len(self.df):
                print("length don't match")
                return
            # train_df = self.df[[self.label_name, self.pred_col_name]].join(_df)
            train_df = self.df.join(_df)
        # print current score
        if self.score is not None:
            print('The current score is:', self.eval_metrics(self.label, self.score))
        def _cv(_random_state):
            # ModelSaver
            MS = {'pred_train':[],
                  'pred_val':[],
                  'model_lst':[]}
            kf = KFold(n_splits = n_splits, random_state=_random_state, shuffle=True)
            for train_ix, val_ix in kf.split(train_df):
                train_x, train_x_val,train_y,train_y_val = train_test_split(train_df.loc[train_ix,:].drop([self.label_name],axis=1),
                                                                            train_df.loc[train_ix,:][self.label_name].astype(int),
                                                                            test_size=0.3,
                                                                            random_state=_random_state)
                val_x = train_df.loc[val_ix,:].drop([self.label_name],axis=1)
                lgb_clf = self.lgb_model(lgb_params, train_x, train_y, train_x_val, train_y_val)
                MS['model_lst'].append(lgb_clf)
                MS['pred_train'].append([lgb_clf.predict_proba(val_x)[:,1], val_ix])
                MS['pred_val'].append(lgb_clf.predict_proba(self.val.drop(self.label_name,axis=1))[:,1])
            MS['pred_val'] = self.get_avg( MS['pred_val'] )
            _temp_df = pd.DataFrame(index=train_df.index)
            _temp_df['pred'] = 0
            for item in MS['pred_train']:
                _temp_df['pred'].loc[item[1]] = item[0]
            MS['pred_train'] = _temp_df.pred.tolist()
            if is_print:
                print('The valid lgb score is: %.4f, The train lgb score is: %.4f'
                      %(self.eval_metrics(self.val[self.label_name], MS['pred_val']), 
                        self.eval_metrics(self.df[self.label_name],  MS['pred_train'])))
            return MS
        if self.MS_arr is None:
            MS_arr = []
        else:
            MS_arr = self.MS_arr
        for ix_cv in range(round_cv):
            if is_print:
                print('Round %d'%(ix_cv+1))
            MS = _cv(self.random_state + ix_cv)
            MS_arr.append(MS)
        self.MS_arr = MS_arr
        self.get_mean_score() 
        
    
    
    def get_result(self, test):
        pred_lst = []
        for MS in self.MS_arr:
            for _model in MS['model_lst']:
                pred_lst.append(_model.predict_proba(test)[:,1])
        return self.get_avg( pred_lst )
        
    
    def get_mean_score(self):
        if self.is_print:
            print('Calculating the mean score...')
        self.my_score = []
        lst_val = self.get_avg([self.MS_arr[i]['pred_val'] for i in range(len(self.MS_arr))])
        lst_train = self.get_avg([self.MS_arr[i]['pred_train'] for i in range(len(self.MS_arr))])
        val_mean = self.eval_metrics(self.val[self.label_name], lst_val)
        train_mean = self.eval_metrics(self.df[self.label_name],  lst_train)
        if self.is_print:
            print('lgb val: %.4f, train: %.4f'%(val_mean, train_mean))
        self.my_score.append(val_mean)

            
    def get_avg(self, lst):
        _len = len(lst)
        for _i in range(_len):
            if type(lst[_i]) == list:
                lst[_i] = np.array(lst[_i])
            if _i == 0:
                _temp = lst[_i] * (1 / _len)
            else:
                _temp += lst[_i] * (1 / _len)
        return _temp
        
    
    def lgb_model(self, lgb_params, train_x, train_y, train_x_val, train_y_val):
        _params = {'boosting_type':'gbdt', 'num_leaves':200, 
                   'reg_alpha':0., 'reg_lambda':1, 
                   'n_estimators':2000, 'objective':'binary',
                   'subsample':0.7, 'colsample_bytree':0.7, 
                   'learning_rate':0.1, 'min_child_weight':1}
        if lgb_params is not None:
            for _key in lgb_params.keys():
                _params[_key] = lgb_params[_key]
        clf = lgb.LGBMClassifier(boosting_type=_params['boosting_type'], 
                                 num_leaves=_params['num_leaves'], 
                                 reg_alpha=_params['reg_alpha'], 
                                 reg_lambda=_params['reg_lambda'], 
                                 n_estimators=_params['n_estimators'], 
                                 objective=_params['objective'], 
                                 subsample=_params['subsample'], 
                                 colsample_bytree=_params['colsample_bytree'], 
                                 learning_rate=_params['learning_rate'], 
                                 min_child_weight=_params['min_child_weight'],
                                 feature_fraction=1)
        clf.fit(train_x, train_y, eval_set=[(train_x_val, train_y_val)], eval_metric='auc',
                early_stopping_rounds=100, verbose=False)
        return clf
        
