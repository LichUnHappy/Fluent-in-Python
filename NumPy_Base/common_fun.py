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
# def datestr2num(s):
#     return datetime.strptime(s.decode('ascii'), "%Y-%m-%d").date().weekday()

# dates, opens, high, low, close = np.loadtxt('aapl.csv', delimiter=',', usecols=(0,1,2,3,4), converters={0:datestr2num}, unpack=True)

# close = close[:16]
# dates = dates[:16]

# first_Monday = np.ravel(np.where(dates == 0))[0]
# # print("This is the 1st Monday: " + str(first_Monday))

# last_Friday = np.ravel(np.where(dates == 4))[-1]
# # print("This is the last Friday: " + str(last_Friday)) 

# week_indices = np.arange(first_Monday, last_Friday + 1)
# print("Weeks indices initial is " + str(week_indices))

# week_indices = np.split(week_indices, 3)
# print("Weeks indices after split " + str(week_indices))

# def summarize(a, o, h, l, c):
#     monday_open = o[a[0]]
#     week_high = np.max( np.take(h, a))
#     week_low = np.min(np.take(l,a))
#     friday_close = c[a[-1]]

#     return("APPL "+ str(monday_open) + str(week_high) + str(week_low) + str(friday_close))

# weeksummary = np.apply_along_axis(summarize, 1, week_indices, opens, high, low, close)
# print(str(weeksummary))

# ATR(average true range)
# h, l, c = np.loadtxt('aapl.csv', delimiter=',', usecols=(2, 3, 4),
# unpack=True)

# N = 5
# h = h[-N:]
# l = l[-N:]

# print("len(h)", len(h), "len(l)", len(l))
# print("Close", c)
# previousclose = c[-N -1: -1]

# print("len(previousclose)", len(previousclose))
# print("Previous close", previousclose)
# truerange = np.maximum(h - l, h - previousclose, previousclose - l)
# print("True range", truerange)
# atr = np.zeros(N)
# atr[0] = np.mean(truerange)

# for i in range(1, N):
#     atr[i] = (N - 1) * atr[i - 1] + truerange[i]
#     atr[i] /= N
# print("ATR", atr)

# Simple moving average
import matplotlib.pyplot as plt

# N = 300
# weights = np.ones(N) / N
# print("Weights", weights)

# c = np.loadtxt('aapl.csv', delimiter=',', usecols=(3), unpack=True)
# # convolove
# sma = np.convolve(weights, c)[N-1:-N+1]
# t = np.arange(N - 1, len(c))

# plt.plot(t, c[N-1:], lw=1.0, label="Data")
# plt.plot(t, sma, '--', lw=2.0, label="Moving average")

# plt.title("5 Day Moving Average")
# plt.xlabel("Days")
# plt.ylabel("Price ($)")
# plt.grid()
# plt.legend()
# plt.show()

# exponential moving average
# x = np.arange(5)
# print("Exp", np.exp(x))
# print("Linspace", np.linspace(-1, 0, 5))

# # Calculate weights
# N = 5
# weights = np.exp(np.linspace(-1., 0., N))

# # Normalize weights
# weights /= weights.sum()
# print("Weights", weights)

# c = np.loadtxt('aapl.csv', delimiter=',', usecols=3, unpack=True)
# ema = np.convolve(weights, c)[N-1:-N+1]
# t = np.arange(N - 1, len(c))
# plt.plot(t, c[N-1:], lw=1.0, label='Data')
# plt.plot(t, ema, '--', lw=2.0, label='Exponential Moving Average')
# plt.title('5 Days Exponential Moving Average')
# plt.xlabel('Days')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.grid()
# plt.show()

# N = 5
# weights = np.ones(N) / N
# print("Weights", weights)

# c = np.loadtxt('aapl.csv', delimiter=',', usecols=(3), unpack=True)
# sma = np.convolve(weights, c)[N-1:-N+1]

# deviation = []
# C = len(c)
# for i in range(N - 1, C):
#     if i + N < C:
#         dev = c[i: i + N]
#     else:
#         dev = c[-N:]

#     averages = np.zeros(N)
#     averages.fill(sma[i - N - 1])
#     dev = dev - averages
#     dev = dev ** 2
#     dev = np.sqrt(np.mean(dev))
#     deviation.append(dev)

# deviation = 2 * np.array(deviation)
# print(len(deviation), len(sma))

# upperBB = sma + deviation
# lowerBB = sma - deviation
# c_slice = c[N-1:]
# between_bands = np.where((c_slice < upperBB) & (c_slice > lowerBB))
# print(lowerBB[between_bands])
# print(c[between_bands])
# print(upperBB[between_bands])

# between_bands = len(np.ravel(between_bands))
# print("Ratio between bands", float(between_bands)/len(c_slice))

# t = np.arange(N - 1, C)
# plt.plot(t, c_slice, lw=1.0, label='Data')
# plt.plot(t, sma, '--', lw=2.0, label='Moving Average')
# plt.plot(t, upperBB, '-.', lw=3.0, label='Upper Band')
# plt.plot(t, lowerBB, ':', lw=4.0, label='Lower Band')
# plt.title('Bollinger Bands')
# plt.xlabel('Days')
# plt.ylabel('Price ($)')
# plt.grid()
# plt.legend()
# plt.show()

# linear model
# N = 5
# c = np.loadtxt('aapl.csv', delimiter=',', usecols=(3), unpack=True)
# b = c[-N:]
# b = b[::-1]
# print("b", b)
# A = np.zeros((N, N), float)
# print("Zeros N by N", A)
# for i in range(N):
#     A[i, ] = c[-N - 1 - i: - 1 - i]
# print("A", A)

# (x, residuals, rank, s) = np.linalg.lstsq(A, b)
# print(x, residuals, rank, s)

# print("Predict Now")
# print(np.dot(b, x))

# Trend Lines
def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    
    return np.linalg.lstsq(A, y)[0]

# Determine pivots
h, l, c = np.loadtxt('aapl.csv', delimiter=',', usecols=(2,3,4),
unpack=True)
pivots = (h + l + c) / 3
print("Pivots", pivots)

# Fit trend lines
t = np.arange(len(c))
sa, sb = fit_line(t, pivots - (h - l))
ra, rb = fit_line(t, pivots + (h - l))
support = sa * t + sb
resistance = ra * t + rb
condition = (c > support) & (c < resistance)
print("Condition", condition)
between_bands = np.where(condition)
print(support[between_bands])
print(c[between_bands])
print(resistance[between_bands])
between_bands = len(np.ravel(between_bands))
print("Number points between bands", between_bands)
print("Ratio between bands", float(between_bands)/len(c))
print("Tomorrows support", sa * (t[-1] + 1) + sb)
print("Tomorrows resistance", ra * (t[-1] + 1) + rb)
a1 = c[c > support]
a2 = c[c < resistance]
print("Number of points between bands 2nd approach" ,len(np.
intersect1d(a1, a2)))

# Plotting
plt.plot(t, c, label='Data')
plt.plot(t, support, '--', lw=2.0, label='Support')
plt.plot(t, resistance, '-.', lw=3.0, label='Resistance')
plt.title('Trend Lines')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.grid()
plt.legend()
plt.show()