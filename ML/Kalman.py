import numpy as np
import matplotlib.pyplot as plt

class Kalman:
    def __init__(self, q, v, r, p, x, time):
        self.q=q
        self.r=r
        self.p=p
        self.x=x
        self.v=v
        self.lastUpdate = time
        self.k=0.0

    def update(self, measurement, time):
        A = self.dFunc(self.lastUpdate)
        H = A*(time - self.lastUpdate)
        x_last =self.x+H+ self.v
        P_last = H*self.p*H+self.q
        self.k = (P_last*H)/(H*P_last*A + self.r)
        self.x = x_last + self.k*(measurement-H*x_last)
        self.p = (1 - self.k * H) * P_last
        self.lastUpdate = time

    def func(self, X):
        f = 1
        return 4 * np.core.sin(2 * np.pi * f * X)

    def dFunc(self, X):
        f = 1
        return (8*np.pi*f*np.core.cos(2 * np.pi * f * X))

def func(X):
    f = 1
    return 4*np.core.sin(2*np.pi*f*X)
def dFunc(X):
    f = 1
    return 4*np.core.cos(2*np.pi*f*X)

klm = Kalman(0, 0, 1, 0.05, 0, 0)
t = np.arange(0, 10, 0.01)
y = klm.func(t)
ynoise = y+((np.random.rand(y.size)*1000)%2-1)
yfilter = np.zeros(t.size)

for i in range(0, t.size):
    klm.update(ynoise[i], t[i])
    yfilter[i]=klm.x


plt.plot(t, y, 'b')
plt.plot(t, ynoise, 'g')
plt.plot(t, yfilter, 'r')
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('Kalman Filter')
plt.grid(True)
plt.show()


