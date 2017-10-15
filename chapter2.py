# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:59:52 2017

@author: raceh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  # 统计相关的库

def acf(data, m=10):
    '''
    计算m个自相关系数(加上lag=0则是m+1个)
    '''
    acf, q, p = sm.tsa.acf(data, nlags=m, qstat=True) #计算自相关系数、Ljung-Box Q-Statistic及对应的p值
    out = np.c_[range(1,11), acf[1:], q, p]
    output = pd.DataFrame(out, columns=['lag', "ACF", "Q", "P-value"])
    output = output.set_index('lag')
    print(output)


def adj_r2(residual, series):
    '''
    计算拟合优度
    '''    
    score = 1 - residual.var() / series.var() #调整后的拟合优度，越接近1拟合效果越好
    print('拟合优度：', score)


def predict(model, start, end, original):
    '''
    进行预测并画图
    '''
    predicts = model.predict(start, end, dynamic=True) #预测
    comp = pd.DataFrame()
    comp['original'] = original
    comp['predict'] = predicts
    print(comp)
    comp.plot()
    plt.show()

def acf_test():
    '''
    计算自相关系数
    '''
    data = pd.read_csv('m-ibm3dx2608.txt', sep=r'\s+', header=0) #任意空白符为分隔符
    ibm = data['ibmrtn']
    vw = data['vwrtn']
    ibmr = np.log(ibm + 1) #对数收益率
    vwr = np.log(vw + 1) #对数收益率
    
    acf(ibm, 10)
    acf(ibmr, 10)
    acf(vw, 10)
    acf(vwr, 10)

def ar_ma_test():
    '''
    AR(p)模型的特征根及平稳性检验
    '''
    gnp = pd.read_csv('dgnp82.txt', header=None) # Load data
    data = gnp[0].values #注意下面函数的参数要用ndarray，不能用dataframe
    
    model = sm.tsa.AR(data).fit() #拟合AR模型
    print(len(model.roots)) #拟合的阶数是14
    
    #比较图
    plt.figure(figsize=(10, 4))
    plt.plot(gnp, 'b', label='GNP growth rate')
    plt.plot(model.fittedvalues, 'r', label='AR model')
    plt.legend()
    plt.show()
    
    #画出模型的特征根来检验平稳性
    pi, sin, cos = np.pi, np.sin, np.cos
    r1 = 1
    theta = np.linspace(0, 2*pi, 360)
    x1 = r1 * cos(theta)
    y1 = r1 * sin(theta)
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, 'k')  #画单位圆
    roots = 1 / model.roots  #注意，这里m1.roots是计算的特征方程的解，特征根应该取倒数
    for i in range(len(roots)):
        plt.plot(roots[i].real, roots[i].imag, '.r', markersize=8)  #画特征根
    plt.show()
    
    #检验残差序列是否白噪声
    #注意fittedvalues指模型截尾之后的部分
    residual = model.fittedvalues - data[14:] #残差
    plt.figure(figsize=(10, 6))
    plt.plot(residual, 'r', label='residual error')
    plt.legend(loc=0)
    plt.show()
    acf(residual) #计算自相关系数及p-value
    
    
    '''
    AR(p)模型的定阶和应用
    '''
    #画出PACF（偏自相关函数）
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data, ax=ax1)
    plt.show()
    
    #使用信息准则定阶
    aic_list = []
    bic_list = []
    hqic_list = []
    for i in range(1, 11):  #从1阶开始算
        order = (i, 0)  #这里使用了ARMA模型，order代表了模型的(p,q)值，我们令q始终为0，就只考虑了AR情况。
        tmp = sm.tsa.ARMA(data, order).fit() #使用指定阶数AR模型拟合
        aic_list.append(tmp.aic) #赤池信息量 akaike information criterion
        bic_list.append(tmp.bic) #贝叶斯信息量 bayesian information criterion
        hqic_list.append(tmp.hqic) #hannan-quinn criterion
    
    #选择AIC值最小的阶数
    plt.figure(figsize=(15, 6))
    plt.plot(aic_list, 'r', label='aic value')
    plt.plot(bic_list, 'b', label='bic value')
    plt.plot(hqic_list, 'k', label='hqic value')
    plt.legend(loc=0)
    plt.show()
    
    #拟合优度及预测
    train = data[:-10]
    test = data[-10:] #留下最后十个数据进行预测
    
    order = (2, 0) #使用aic定阶
    model = sm.tsa.ARMA(train, order).fit() #拟合AR模型
    adj_r2(model.fittedvalues - train, train) #拟合优度
    predict(model, 166, 175, test) #预测
    
    
    '''
    MA(q)模型的定阶和应用
    '''
    #使用MA(q)模型的ACF函数q步截尾来判断模型阶次
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_acf(data, ax=ax1)
    plt.show()
    
    order = (0, 2) #由于sm.tsa中没有单独的MA模块，我们利用ARMA模块测试MA模型，只要将其中AR的阶p设为0
    model = sm.tsa.ARMA(train, order).fit() #拟合MA模型
    adj_r2(model.fittedvalues - train, train) #拟合优度
    predict(model, 166, 175, test) #预测
    
    
    '''
    ARMA(p, q)模型的定阶和应用
    '''
    #通过观察PACF和ACF截尾，分别判断p、q的值
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=30, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=30, ax=ax2)
    plt.show() #观察得到acf的截尾是2，pacf的截尾是1
    
    #使用信息准则定阶
    order_aic = sm.tsa.arma_order_select_ic(data, max_ar=6, max_ma=4, ic='aic')['aic_min_order']  # AIC
    order_bic = sm.tsa.arma_order_select_ic(data, max_ar=6, max_ma=4, ic='bic')['bic_min_order']  # BIC
    order_hqic = sm.tsa.arma_order_select_ic(data, max_ar=6, max_ma=4, ic='hqic')['hqic_min_order'] # HQIC
    print(order_aic) #(2,2)
    print(order_bic) #(0,2)
    print(order_hqic) #(0,2)
    
    model = sm.tsa.ARMA(train, order_aic).fit() #使用AIC准则求解的模型阶次来拟合ARMA模型
    adj_r2(model.fittedvalues - train, train) #拟合优度
    predict(model, 166, 175, test) #预测


'''
ARIMA(p, d, q)模型的定阶和应用
'''
temp = pd.read_csv('q-gdp4708.txt', sep=r'\s+', header=0) #美国季度GDP
gdp = np.log(temp['gdp']) #对数数据
data = gdp.values #注意下面函数的参数要用ndarray，不能用dataframe
gdp.plot(figsize=(15, 5))
plt.show()

#ADF单位根检验
t = sm.tsa.stattools.adfuller(data)
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output) #p-value大于显著性水平，原假设序列具有单位根（即非平稳）成立

#将序列进行1次差分后再次检验
data_diff = gdp.diff().values
gdp.diff().plot(figsize=(15, 5))
plt.show()
data_diff = data_diff[1:] #差分后第一个值为NaN舍去
t = sm.tsa.stattools.adfuller(data_diff)
print('p-value:', t[1])

#ARIMA(p,d,q)模型定阶
#使用ACF和PACF
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_diff, lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_diff, lags=30, ax=ax2)
plt.show()
'''
#使用AIC
order_aic = sm.tsa.arma_order_select_ic(data_diff, max_ar=6, max_ma=4, ic='aic')['aic_min_order']
print(order_aic) #(6, 1)
'''
#要建立的ARIMA模型阶次(p, d, q) = (6, 1, 1)
#对差分后序列建立ARMA模型
order = (6, 1)
train = data_diff[:-10]
test = data_diff[-10:]

model = sm.tsa.ARMA(train, order).fit()
plt.figure(figsize=(15, 5))
#查看拟合情况
plt.plot(model.fittedvalues, label='fitted value')
plt.plot(train, label='real value')
plt.legend(loc=0)
adj_r2(model.fittedvalues - train, train) #拟合优度
predict(model, 237, 246, test) #预测

