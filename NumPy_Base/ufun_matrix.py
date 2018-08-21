import numpy as np
import matplotlib.pyplot as plt

# A = np.mat('1 2 3; 4 5 6; 7 8 9')
# print("Creation from string", A)
# print("transpose A", A.T)
# print("Inverse A", A.I)
# print("Check Inverse", A * A.I)
# print("Creation from array", np.mat(np.arange(9).reshape(3, 3)))

# A = np.eye(2)
# print("A", A)
# B = 2 * A
# print("B", B)
# print("Compound matrix\n", np.bmat("A B; A B"))

# def ultimate_answer(a):
#     result = np.zeros_like(a)
#     result.flat = 42
#     return result

# ufunc = np.frompyfunc(ultimate_answer, 1, 1)

# print("The answer", ufunc(np.arange(4)))
# print("The answer", ufunc(np.arange(4).reshape(2, 2)))

# a = np.arange(-4, 4)
# print("Remainder", np.remainder(a, 2))
# print("Mod", np.mod(a, 2))
# print("% operator", a % 2)
# print("Fmod", np.fmod(a, 2))

# F = np.matrix([[1, 1], [1, 0]])
# print("F", F)
# print("8th Fibonacci", (F ** 7)[0, 0])

# n = np.arange(1, 9)
# sqrt5 = np.sqrt(5)
# phi = (1 + sqrt5)/2
# fibonacci = np.rint((phi**n - (-1/phi)**n)/sqrt5)
# print("Fibonacci", fibonacci)

# import matplotlib.pyplot as plt
# a = 9
# b = 8
# t = np.linspace(-np.pi, np.pi, 201)
# x = np.sin(a * t + np.pi/2)
# y = np.sin(b * t)
# plt.plot(x, y)
# plt.title('Lissajous curves')
# plt.grid()
# plt.show()

# t = np.linspace(-np.pi, np.pi, 201)
# k = np.arange(1, 10)
# k = 2 * k - 1
# f = np.zeros_like(t)

# for i, ti in enumerate(t):
#     f[i] = np.sum(np.sin(k * ti)/k)

# f = (4 / np.pi) * f

# plt.plot(t, f)
# plt.title('Square wave')
# plt.grid()
# plt.show()

# t = np.linspace(-np.pi, np.pi, 201)
# k = np.arange(1, 99)
# f = np.zeros_like(t)

# for i, ti in enumerate(t):
#     f[i] = np.sum(np.sin(2 * np.pi * k * ti)/k)
# f = (-2 / np.pi) * f
# plt.plot(t, f, lw=1.0, label='Sawtooth')
# plt.plot(t, np.abs(f), '--', lw=2.0, label='Triangle')
# plt.title('Triangle and sawtooth waves')
# plt.grid()
# plt.legend()
# plt.show()

