# DF-Early-warning-of-the-wind-power-system Rank2
# DF风机叶片开裂预警单赛道第二名，总决赛二等奖方案分享
注意：本方案包含了
+ 1> 代码审核时提交的部分（能够直接运行）
+ 2> 做线下训练时的代码（包含各种模型）

# 运行方法：直接运行main.py
1.环境配置和依赖库：
+ python3
+ lightgbm
+ multiprocessing
+ tqdm

2.特征说明：
+ 基本统计特征：每个column对应的的mean, max, min, var, ptp, median
+ 特征总数：75 * 6 = 450

3.数据预处理：
+ 1> 把全0行数据替换为均值（当然也可以直接去掉）
+ 2> 先把所有数据除以均值，然后再做相关统计（有利于产生更多有意义的特征组合）

4.训练模型：
+ 1> lightgbm
+ 2> KNN
+ 3> SVM
