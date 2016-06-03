import numpy as np
import matplotlib.pyplot as plt

def markovChain(s0, T, it):
    s = s0.copy()
    result = np.zeros(it+1)
    result[0]=s[0]
    for i in range(1, it+1):
        s = s.dot(T)
        result[i]=s[0]
    return result

s0 = np.array([1, 0])
s0inv = np.array([0.015, 0.985])
T = np.array([[0.35, 0.65],[0.015, 0.985]])
n = np.arange(0, 16, 1)

gen15 = markovChain(s0, T, 15)
gen15inv = markovChain(s0inv, T, 15)

plt.plot(n, gen15, 'b')
plt.plot(n, gen15inv, 'r')
plt.xlabel('n - generations')
plt.ylabel('probability')
plt.title('Virus Generations')
plt.grid(True)
plt.show()

