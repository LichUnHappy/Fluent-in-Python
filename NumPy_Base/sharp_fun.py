import numpy as np
import matplotlib.pyplot as plt


# 相关性 
# bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
# bhp_returns = np.diff(bhp) / bhp[ : -1]
# vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,),
# unpack=True)
# vale_returns = np.diff(vale) / vale[ : -1]
# covariance = np.cov(bhp_returns, vale_returns)
# print("Covariance", covariance)
# print("Covariance diagonal", covariance.diagonal())
# print("Covariance trace", covariance.trace())
# print(covariance/ (bhp_returns.std() * vale_returns.std()))
# print("Correlation coefficient", np.corrcoef(bhp_returns, vale_returns))
# difference = bhp - vale
# avg = np.mean(difference)
# dev = np.std(difference)
# print("Out of sync", np.abs(difference[-1] - avg) > 2 * dev)
# t = np.arange(len(bhp_returns))
# plt.plot(t, bhp_returns, lw=1, label='BHP returns')
# plt.plot(t, vale_returns, '--', lw=2, label='VALE returns')
# plt.title('Correlating arrays')
# plt.xlabel('Days')
# plt.ylabel('Returns')
# plt.grid()
# plt.legend(loc='best')
# plt.show()

# 多项式 plyfit

# bhp=np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
# vale=np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)
# t = np.arange(len(bhp))
# poly = np.polyfit(t, bhp - vale, 5)
# print("Polynomial fit", poly)
# print("Next value", np.polyval(poly, t[-1] + 1))
# print("Roots", np.roots(poly))
# der = np.polyder(poly)
# print("Derivative", der)
# print("Extremas", np.roots(der))
# vals = np.polyval(poly, t)
# print(np.argmax(vals))
# print(np.argmin(vals))
# plt.plot(t, bhp - vale, label='BHP - VALE')
# plt.plot(t, vals, '--', label='Fit')
# plt.title('Polynomial fit')
# plt.xlabel('Days')
# plt.ylabel('Difference ($)')
# plt.grid()
# plt.legend()
# plt.show()

# 经成交额 on balance volume
# c, v=np.loadtxt('BHP.csv', delimiter=',', usecols=(6, 7), unpack=True)
# change = np.diff(c)
# print("Change", change)
# signs = np.sign(change)
# print("Signs", signs)
# pieces = np.piecewise(change, [change < 0, change > 0], [-1, 1])
# print("Pieces", pieces)
# print("Arrays equal?", np.array_equal(signs, pieces))
# print("On balance volume", v[1:] * signs)

# simulation
# o, h, l, c = np.loadtxt('BHP.csv', delimiter=',', usecols=(3, 4, 5, 6), unpack=True)

# def calc_profit(open, high, low, close):
#     #buy just below the open
#     buy = open * 0.999
#     # daily range
#     if low < buy < high:
#         return (close - buy)/buy
#     else:
#         return 0

# func = np.vectorize(calc_profit)

# profits = func(o, h, l, c)
# print("Profits", profits)

# real_trades = profits[profits != 0]
# print("Number of trades", len(real_trades), round(100.0 * len(real_trades)/len(c), 2), "%")
# print("Average profit/loss %", round(np.mean(real_trades) * 100, 2))

# winning_trades = profits[profits > 0]
# print("Number of winning trades", len(winning_trades), round(100.0 *
# len(winning_trades)/len(c), 2), "%")
# print("Average profit %", round(np.mean(winning_trades) * 100, 2))

# losing_trades = profits[profits < 0]
# print("Number of losing trades", len(losing_trades), round(100.0 *
# len(losing_trades)/len(c), 2), "%")
# print("Average loss %", round(np.mean(losing_trades) * 100, 2))

# 数据smooth
N = 8
weights = np.hanning(N)
print("Weights", weights)

bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
bhp_returns = np.diff(bhp) / bhp[ : -1]
smooth_bhp = np.convolve(weights/weights.sum(), bhp_returns)[N-1:-N+1]

vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,),unpack=True)
vale_returns = np.diff(vale) / vale[ : -1]
smooth_vale = np.convolve(weights/weights.sum(), vale_returns)[N-1:-N+1]

K = 8
t = np.arange(N - 1, len(bhp_returns))
poly_bhp = np.polyfit(t, smooth_bhp, K)
poly_vale = np.polyfit(t, smooth_vale, K)
poly_sub = np.polysub(poly_bhp, poly_vale)

xpoints = np.roots(poly_sub)
print("Intersection points", xpoints)

reals = np.isreal(xpoints)
print("Real number?", reals)

xpoints = np.select([reals], [xpoints])
xpoints = xpoints.real
print("Real intersection points", xpoints)
print("Sans 0s", np.trim_zeros(xpoints))

plt.plot(t, bhp_returns[N-1:], lw=1.0, label='BHP returns')
plt.plot(t, smooth_bhp, lw=2.0, label='BHP smoothed')
plt.plot(t, vale_returns[N-1:], '--', lw=1.0, label='VALE returns')
plt.plot(t, smooth_vale, '-.', lw=2.0, label='VALE smoothed')
plt.title('Smoothing')
plt.xlabel('Days')
plt.ylabel('Returns')
plt.grid()
plt.legend(loc='best')
plt.show()

