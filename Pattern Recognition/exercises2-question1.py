import matplotlib.pyplot as plt
import numpy as np

points = []

def triangular(x, a, b, c):
    points.clear()
    if (a<=b)<=c :
        if x<=a :
            points.extend(0)
        elif (a<=x)<=b:
            points.extend((x-a)/(b-a))
        elif (b<=x)<=c:
            points.extend((c-x)/(c-b))
        elif c<=x :
            points.extend(0)
    return 0

def trapezoid(x,a,b,c,d):
    y = np.where(x <= a, 0, x)
    y = np.where(np.logical_and(a <= x, x <= b), (x - a) / (b - a), y)
    y = np.where(np.logical_and(b <= x, x <= c), 1, y)
    y = np.where(np.logical_and(c <= x, x <= d), (d - x) / (d - c), y)
    y = np.where(d <= x, 0, y)
    return y

def bell(x,a,b,c):
    points.clear()
    if a>0 :
        points.extend(1/(1+((x-c)/a)**(2*b)))
    return 0

def gaussian(x,c,nu):
    if nu>0:
        points.extend(np.exp((-1/2)*((x-c)/nu)**2))
    return 0

x=[-8,-5,0,5,8]
for i in x:

    k=trapezoid(i,-20,-15,-6,-3)
    points.append(k)

print(points)
plt.figure(figsize=(16,4))
plt.plot(points)
plt.show()
