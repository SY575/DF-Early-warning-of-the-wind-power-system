# -*- coding: utf-8 -*-

'''
运行环境：
win10 12G内存
lightgbm 2.1.2

运行方法：
1.修改config中的文件路径
2.运行main.py（本文件）

代码说明：
由于原解决方案中包括一些我们还未发表的论文中的重要方法，
因此我们把其中一部分方法做了适当替换，但是还是基本能够复现出我们的最高成绩。
望主办方能够理解！
'''

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
sys.path.append('./utils/')
from utils import get_feature_ori, get_feature, get_baseline, get_multi_label, get_new_label
import config
import pandas as pd

# 生成特征文件
data_ori = get_feature_ori.run()
data = get_feature.run()


result_1 = get_baseline.run(data_ori.copy())

result_2 = get_multi_label.run(data.copy(), result_1.copy())

result_3 = get_new_label.run(data.copy(), result_2.copy())

# 存储
result = pd.read_csv(config.SUBMIT_SAMPLE_PATH)
result = pd.merge(result[['id']], result_3[['id', 'ret']], on='id', how='left')
result['ret'] = result['ret'].astype(int)

result.to_csv(config.SAVE_PATH, index=False)

