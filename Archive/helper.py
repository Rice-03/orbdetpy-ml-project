import numpy as np

x = np.arange(0, 10) ** 2
y = np.diff(x)
print("x:", x)
print("y:", y)

a = np.random.rand(15, 4)
b = np.diff(a[:,2])
print("a:", a)
print("b:", b)