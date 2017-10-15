# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:51:10 2017

@author: raceh
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp #单样本t检验，检验均值是否等于假设均值
#from scipy.stats import ttest_ind #两个独立样本的t检验，目的是判断两个样本均数所对应的总体均值是否有差别

def fin_stats(data):
    '''
    计算数据均值，标准差，偏度，超额峰度，最小值和最大值
    '''
    tplt = '{0:^20}\t{1:^20}\t{2:^20}\t{3:^20}\t{4:^20}\t{5:^20}'
    print(tplt.format('mean', 'std', 'skew', 'kurt', 'min', 'max'))
    print(tplt.format(data.mean(), data.std(), data.skew(), data.kurt(), data.min(), data.max()))

def diff_test():
    data = pd.Series([1, 1.1, 1.21, 1.331, 1.4641]) #10%增长率序列
    data_diff = data.diff()
    growth = data_diff / data.shift(1) #原序列右移一位
    print(growth.dropna())


#dstock = pd.read_csv('d-3stocks9908.txt', sep=r'\s+', header=0) #任意空白符为分隔符
dstock = pd.read_csv('d-3stock.txt', sep=r'\s+', header=None) #任意空白符为分隔符

#百分数形式表示
#axp = dstock['axp'] * 100
#cat = dstock['cat'] * 100
#sbux = dstock['sbux'] * 100
axp = dstock[1] * 100
cat = dstock[2] * 100
sbux = dstock[3] * 100

fin_stats(axp)
fin_stats(cat)
fin_stats(sbux)

#简单收益率转对数收益率
axpr = 100 * np.log(dstock[1] + 1)
catr = 100 * np.log(dstock[2] + 1)
sbuxr = 100 * np.log(dstock[3] + 1)

fin_stats(axpr)
fin_stats(catr)
fin_stats(sbuxr)

#one sample t-test
#p值均大于0.05
print(ttest_1samp(axpr, popmean=0))
print(ttest_1samp(catr, popmean=0))
print(ttest_1samp(sbuxr, popmean=0))
