import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
import math


def rosenbrock(x):
    sum = 0
    for i in range(0,x.size-1):
        value =  100*math.pow(x[i+1]-x[i]*x[i], 2)+math.pow(x[i]-1, 2)
        sum += value
    return sum


def gradDescent(fun, param, it):
    x = param.copy()
    eps = 0.0000000001
    #last_val = fun(x)
    for i in range(0,it):
        #step = 0.000009995
        t = 1;
        count = 0
        left = fun(x)
        right = fun(x)
        while (left >= right and t>0):
            t =t*0.1511
            y=x - t * numPartDiff(fun, x, eps)
            left = fun(y)
            count = count + 1
        #print(t)
        x = x - t * numPartDiff(fun, x, eps)

    return x

def numDiff(fun, params, i, eps):
    x_plus = params.copy()
    x_minus = params.copy()
    x_plus[i]=x_plus[i]+eps
    x_minus[i]=x_minus[i]-eps
    return (fun(x_plus)-fun(params))/(eps)

def numPartDiff(fun, params, eps):
    J = np.zeros(params.size)
    for i in range(0, params.size):
        J[i]=numDiff(fun, params, i, eps)
    return J


x = np.arange(1, 21, 1)

print(x)
print(rosenbrock(x))
#print(numDiff(rosenbrock, x, 0, 0.1))
#print(numPartDiff(rosenbrock, x, 0.1))
args = gradDescent(rosenbrock, x, 100)
print(args)
print(rosenbrock(args))
#print(minimize(rosenbrock, x))



#plt.xlabel('xi - values')
#plt.ylabel('fi - values')
#plt.title('Rosen')
#plt.grid(True)
#plt.show()
