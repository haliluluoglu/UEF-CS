##Exercise 2
#Question 1

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def triangular(x, a, b, c):
    if np.logical_and(a<=b,b<=c):
        if x<=a :
            return 0
        elif np.logical_and(a<=x,x<=b):
            return((x-a)/(b-a))
        elif np.logical_and(b<=x,x<=c):
            return((c-x)/(c-b))
        elif c<=x:
            return 0

def trapezoid(x,a,b,c,d):
    if np.logical_and(np.logical_and(a<=b,b<=c),c<=d):
        if x<=a:
            return 0
        elif np.logical_and(a<=x,x<=b):
            return ((x-a)/(b-a))
        elif np.logical_and(b<=x,x<=c):
            return 1
        elif np.logical_and(c<=x,x<=d):
            return ((d-x)/(d-c))
        elif d<=x:
            return 0

def bell(x,a,b,c):
    if a>0:
        return(1/(1+((x-c)/a)**(2*b)))
    return 0

def gaussian(x,c,nu):
    if nu>0:
        return(np.exp((-1/2)*((x-c)/nu)**2))
    return 0
def printaplot(points):
    plt.plot(points)
    plt.show()

x=[-8,-5,0,5,8]
y=[-3,-1,0,1,3]
z=[0,5,10,15,20]
points = [0]
for i in x:
    k=triangular(i,-20,-15,-6)
    points.append(k)
points.append(0)
printaplot(points)
points.clear()
points.append(0)
for i in y:
    k=triangular(i,-10,-5,0)
    points.append(k)
points.append(0)
printaplot(points)
points.clear()
points.append(0)
for i in z:
    k=triangular(i,0,10,20)
    points.append(k)
points.append(0)
printaplot(points)
points.clear()
points.append(0)

# for i in x:
#     k=trapezoid(i,-20,-15,-6,-2)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in y:
#     k=trapezoid(i,-10,-5,0,5)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in z:
#     k=trapezoid(i,0,10,20,30)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)

# for i in x:
#     k=bell(i,3,6,15)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in y:
#     k=bell(i,10,5,0)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in z:
#     k=bell(i,5,10,20)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)

# for i in x:
#     k=gaussian(i,6,15)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in y:
#     k=gaussian(i,5,2)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)
# for i in z:
#     k=gaussian(i,10,20)
#     points.append(k)
# points.append(0)
# printaplot(points)
# points.clear()
# points.append(0)

#Question 2
#
# def trapezoid(x,a,b,c,d):
#     if np.logical_and(np.logical_and(a<=b,b<=c),c<=d):
#         if x<=a:
#             return 0
#         elif np.logical_and(a<=x,x<=b):
#             return ((x-a)/(b-a))
#         elif np.logical_and(b<=x,x<=c):
#             return 1
#         elif np.logical_and(c<=x,x<=d):
#             return ((d-x)/(d-c))
#         elif d<=x:
#             return 0
#
# def defuzzification(x,a,b):
#     b = b.lower()
#     x = x.ravel()
#     a = a.ravel()
#     n = len(x)
#     assert n == len(a)
#
#     if 'centroid' in b or 'bisector' in b:
#         zero_truth_degree = a.sum() == 0
#         assert not zero_truth_degree
#
#         if 'centroid' in b:
#             return centroid(x, a)
#
#         elif 'bisector' in b:
#             return bisector(x, a)
#
#     elif 'mom' in b:
#         return np.mean(x[a == a.max()])
#
#     elif 'som' in b:
#         tmp = x[a == a.max()]
#         return tmp[tmp == np.abs(tmp).min()][0]
#
#     elif 'lom' in b:
#         tmp = x[a == a.max()]
#         return tmp[tmp == np.abs(tmp).max()][0]
#
#     else:
#         raise ValueError('The input for `b`, %s, was incorrect.' % (b))
#
#
#     def centroid(x, a):
#
#         sum_moment_area = 0.0
#         sum_area = 0.0
#
#         for i in range(1, len(x)):
#             x1 = x[i - 1]
#             x2 = x[i]
#             y1 = a[i - 1]
#             y2 = a[i]
#
#             if not(y1 == y2 == 0.0 or x1 == x2):
#                 if y1 == y2:
#                     moment = 0.5 * (x1 + x2)
#                     area = (x2 - x1) * y1
#                 elif y1 == 0.0 and y2 != 0.0:
#                     moment = 2.0 / 3.0 * (x2-x1) + x1
#                     area = 0.5 * (x2 - x1) * y2
#                 elif y2 == 0.0 and y1 != 0.0:
#                     moment = 1.0 / 3.0 * (x2 - x1) + x1
#                     area = 0.5 * (x2 - x1) * y1
#                 else:
#                     moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
#                     area = 0.5 * (x2 - x1) * (y1 + y2)
#
#                 sum_moment_area += moment * area
#                 sum_area += area
#
#         return sum_moment_area / np.fmax(sum_area,np.finfo(float).eps).astype(float)
#
#
# def bisector(x, a):
#     sum_area = 0.0
#     accum_area = [0.0] * (len(x) - 1)
#
#     for i in range(1, len(x)):
#         x1 = x[i - 1]
#         x2 = x[i]
#         y1 = a[i - 1]
#         y2 = a[i]
#
#         if not(y1 == y2 == 0. or x1 == x2):
#             if y1 == y2:
#                 area = (x2 - x1) * y1
#             elif y1 == 0. and y2 != 0.:
#                 area = 0.5 * (x2 - x1) * y2
#             elif y2 == 0. and y1 != 0.:
#                 area = 0.5 * (x2 - x1) * y1
#             else:
#                 area = 0.5 * (x2 - x1) * (y1 + y2)
#             sum_area += area
#             accum_area[i - 1] = sum_area
#
#     index = np.nonzero(np.array(accum_area) >= sum_area / 2.)[0][0]
#
#
#     if index == 0:
#         subarea = 0
#     else:
#         subarea = accum_area[index - 1]
#     x1 = x[index]
#     x2 = x[index + 1]
#     y1 = a[index]
#     y2 = a[index + 1]
#
#     subarea = sum_area/2. - subarea
#
#     x2minusx1 = x2 - x1
#     if y1 == y2:
#         u = subarea/y1 + x1
#     elif y1 == 0.0 and y2 != 0.0:
#         root = np.sqrt(2. * subarea * x2minusx1 / y2)
#         u = (x1 + root)
#     elif y2 == 0.0 and y1 != 0.0:
#         root = np.sqrt(x2minusx1*x2minusx1 - (2.*subarea*x2minusx1/y1))
#         u = (x2 - root)
#     else:
#         m = (y2-y1) / x2minusx1
#         root = np.sqrt(y1*y1 + 2.0*m*subarea)
#         u = (x1 - (y1-root) / m)
#     return u
