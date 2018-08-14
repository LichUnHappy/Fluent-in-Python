# -*- coding: UTF-8 -*-
import numpy as np

# # 计算均值
# c, v = np.loadtxt('aapl.csv', delimiter=',', usecols=(4,5), unpack=True)

# ## 计算成交量加权平均价格(VWAP， volume-weighted average price)
# vwap = np.average(c, weights=v)
# print("VWAP = "+ str(vwap))

# ## 计算算术平均值
# vmean = np.mean(c)
# print("Vmean = " + str(vmean))

# ## 伪-时间加权平均价格 TWAP(Time-weighted average price)。 简单用arange汉书创建一个从0开始的序列，自然数的个数即为收盘价的个数
# t = np.arange(len(c))[::-1] # 倒排 最近的序列最大
# vtwap = np.average(c, weights=t)
# print(vtwap)

# # 计算最大值 最小值

# h, l = np.loadtxt('aapl.csv', delimiter=',', usecols=(2,3), unpack=True)
# print("Highest: " + str(np.max(h)))
# print("Lowest: " + str(np.min(l)))

# # 极差 ptp函数可计算数组的取值范围， 返回数组元素的最大值与最小值之差
# print("Spread High:" + str(np.ptp(h)))
# print("Spread Low:" + str(np.ptp(l)))

# # 中位数
# c = np.loadtxt('aapl.csv', delimiter=',', usecols=4, unpack=True)
# print(np.median(c))

# # 方差
# print("Variance: " + str(np.var(c)))

# 收益率
## 收盘价分析常常是基于股票收益率和对数收益率的。简单收益率是指相邻两个价格之间的变化率，而对数收益率是指所有价格取对数后两两相减

# c = np.loadtxt('aapl.csv', delimiter=',', usecols=4, unpack=True)

# returns = np.diff(c) / c[:-1]
# print("St D: " + str(returns))
# logreturns = np.diff(np.log(c))
# print("Log D: " + str(logreturns))

# # 筛选正值 where!
# postive_index = np.where(returns > 0)
# print("Indice of + :\n" + str(postive_index))

# # 波动率(volatility) 计算历史波动率时，需要用到对数收益率，年波动率等于对数收益率的标准差除以均值，再初一交易日倒数的平方根，通常交易日取252天。
# annual_volatility = np.std(logreturns) / np.mean(logreturns) /np.sqrt(1. / 12.)
# print("Annual volatility: " + str(annual_volatility)) # userows 限定
# print("Monthly volatility: " + str(annual_volatility * np.sqrt(1./12.)))

# 分析日期
from datetime import datetime

# 星期一 1
# 星期二 2
# 星期三 3
# 星期四 4
# 星期五 5
# 星期六 6
# 星期七 7 不开盘

# def datestr2num(s):
    # return datetime.strptime(s.decode('ascii'), "%Y-%m-%d").date().weekday()

# dates, close= np.loadtxt('aapl.csv', delimiter=',', usecols=(0,4), unpack=True, converters={0:datestr2num})
# 测试
# raw ="1994-01-17"
# dates = datestr2num(raw)

# # 分组存储与比较
# average = np.zeros(5)
# for i in range(5):
#     indices = np.where(dates == i)
#     prices = np.take(close, indices)
#     avg = np.mean(prices)
#     print("Day " + str(i)+ " Prices " + str(prices) + " Average " + str(avg))
#     average[i] = avg

# top = np.max(average)
# print("Highest average " + str(top))
# print("Top day of the week ", + np.argmax(average))
# bottom = np.min(average)
# print("Highest average " + str(bottom))
# print("Bottom day of the week ", + np.argmin(average))

# 汇总数据
## 周汇总
def datestr2num(s):
    return datetime.strptime(s.decode('ascii'), "%Y-%m-%d").date().weekday()

dates, opens, high, low, close = np.loadtxt('aapl.csv', delimiter=',', usecols=(0,1,2,3,4), converters={0:datestr2num}, unpack=True)

close = close[:16]
dates = dates[:16]

first_Monday = np.ravel(np.where(dates == 0))[0]
# print("This is the 1st Monday: " + str(first_Monday))

last_Friday = np.ravel(np.where(dates == 4))[-1]
# print("This is the last Friday: " + str(last_Friday)) 

week_indices = np.arange(first_Monday, last_Friday + 1)
print("Weeks indices initial is " + str(week_indices))

week_indices = np.split(week_indices, 3)
print("Weeks indices after split " + str(week_indices))

def summarize(a, o, h, l, c):
    monday_open = o[a[0]]
    week_high = np.max( np.take(h, a))
    week_low = np.min(np.take(l,a))
    friday_close = c[a[-1]]

    return("APPL "+ str(monday_open) + str(week_high) + str(week_low) + str(friday_close))

weeksummary = np.apply_along_axis(summarize, 1, week_indices, opens, high, low, close)
print(str(weeksummary))