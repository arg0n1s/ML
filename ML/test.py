import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der, curve_fit
import scipy.stats as sp

"""
s0 = np.array([1, 0])
s0alt = np.array([0.015, 0.985])
T = np.array([[0.365, 0.635],[0.015, 0.985]])
print(s0)
print(T)
#s1 = s0.dot(np.linalg.matrix_power(T, 15))
s1 = s0;
s1alt = s0alt
se = np.zeros(16)
seAlt = np.zeros(16)
t = np.arange(0, 16, 1)
se[0] = s0[0]
seAlt[0]=s0alt[0]
for i in range(1, 16):
   s1 = s1.dot(T)
   se[i]=s1[0]
   s1alt = s1alt.dot(T)
   seAlt[i]=s1alt[0]

print(s1)
print(s1alt)

plt.plot(t, se, 'b')
plt.plot(t, seAlt, 'r')
plt.xlabel('n - generations')
plt.ylabel('probability')
plt.title('Virus bla')
plt.grid(True)
plt.show()




"""

def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

def linear_fit(x, a, b):
    return a*x+b

class RunningList:
   def __init__(self, size):
      self.data = []
      self.size = size
      self.count = 0
      self.all = []

   def add(self, x):
      if self.count != self.size:
         self.data.append(x)
         self.count+=1
      else:
         self.data.pop(0)
         self.data.append(x)



   def extrapolate(self):
      if self.count > 2:
        y = np.array(self.data)
        x = np.arange(0, self.count, 1)
        #A = np.array([x, np.ones(x.size)])
        #fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
        fitting_parameters, covariance = curve_fit(linear_fit, x, y)
        a, b = fitting_parameters
        #w = np.linalg.lstsq(A.T, y)[0]
        next_x = self.size
        #x = x.__add__(next_x)
        #y = w[0]*x + w[1]
        next_y = linear_fit(next_x, a, b)
        #next_y = y[y.size-1]
        self.all.append(next_y)
        return next_y, covariance, a
      else:
        self.all.append(self.data[self.count-1])
        return self.data[self.count-1], 0, 0

   def getAll(self):
        self.all.append(self.data[0])
        self.all.append(self.data[1])
        out = np.array(self.all)
        return out

class Kalman:
    def __init__(self, q, r, p, x):
        self.q=q
        self.r=r
        self.p=p
        self.x=x
        self.k=0.0
        self.extra = RunningList(3)
        self.extra.add(x)

    def update(self, measurement):
        x, cov, hk = self.extra.extrapolate()
        self.k = (self.p*hk)/(hk*hk*self.p + self.r)
        self.p = (1-self.k*hk)*self.p
        print("x- : %f- / x: %f / hk: %f / kk: %f / pk: %f" %(self.x, x, hk, self.k, self.p))
        self.x = measurement + self.k*(measurement-x)
        self.extra.add(measurement)

        #self.p = self.p + self.q
        #self.k = self.p / (self.p + self.r)
        #self.x = self.x + self.k * (measurement - self.x)
        #self.x = self.x + self.k * (measurement - self.x)
        #self.p = (1 - self.k) * self.p

class Filter:
    def __init__(self):
        s = np.array([16, 16, 16, 16, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 20, 18, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15])
        s = np.concatenate((s, s))
        s = np.concatenate((s, s))
        #s = np.zeros(100)+20
        self.s = s
        self.t = np.arange(0, s.size, 1)
        self.sn = s+((np.random.rand(s.size)*1000)%2-1)
        self.sf = self.sn.copy()
        self.r = np.std(self.sn)

    def filter(self, q, r, p, sn):
        sf = sn.copy()
        self.kf = Kalman(q, r, p, sf[1])
        for i in range(2, sf.size):
            self.kf.update(sf[i])
            sf[i] = self.kf.x
        return sf

    def costfun(self, params):
        q = params[0];
        r = params[1];
        p = params[2];
        sf = self.filter(q,r,p,self.sn)
        return 1-sp.pearsonr(self.s, sf)[0]

flt = Filter()
s = flt.s
t = flt.t
sn = flt.sn
#goodparams = np.array([-0.189, 575.296, 1247,251])
#sf = flt.filter(goodparams[0], goodparams[1], goodparams[2], flt.sn)
params = np.array([1, 5.5, 0.99])
#pm = minimize(flt.costfun, params)
#p=pm.x
p = params
#print(pm)
print(p)
sfm = flt.filter(p[0], p[1], p[2], flt.sn)
#d = np.abs(s - sfm)
#print(d)
#print(sp.pearsonr(s, sfm))


plt.plot(t, s, 'r')
#plt.plot(t, sn, 'g')
#plt.plot(t, sf, 'b')
plt.plot(t, sfm, 'y')
#plt.plot(t, flt.kf.extra.getAll(), 'b')
plt.axis([0, s.size, 10, 30])
#print("Standard deviation r: %r" %flt.r)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.show()