# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:45:19 2017

@author: raceh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  # 统计相关的库

da = pd.read_csv('m-intc7308.txt', sep=r'\s+', header=0) #任意空白符为分隔符
intc = np.log(da['rtn'] + 1)
plt.plot(intc) #或者调用intc.plot()
plt.show()

