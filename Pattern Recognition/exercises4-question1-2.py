##Exercise 4
#Question 1

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


file = pd.read_csv("C:/Users/Halil/Desktop/Pattern/Exercises/4.exercise/alabama.txt", sep=' ', header=None, names=['Year', 'Number'])
plt.plot(file['Year'], file['Number'])
plt.show()

def linearRegression(x, y):
  if x.ndim == 1:
    x = np.expand_dims(x, axis=-1)
  y = y.ravel()
  nSamples, nFeatures = x.shape

  fPred = lambda x: np.dot(x, 0.1) + 1
  yPred = fPred(x)
  return yPred, fPred

yPred, fPred = linearRegression(file['Year'].values, file['Number'].values)
xNew = np.expand_dims(np.arange(np.max(file['Year']) + 1, 2050 + 1), axis=1)
y_new = fPred(xNew)

x1 = np.expand_dims(file['Year'].values - 1971, axis=1)
x2 = np.expand_dims(np.power(file['Year'].values - 1971, 2),axis=1)
x3 = np.expand_dims(np.power(file['Year'].values - 1971, 3),axis=1)
X = np.concatenate([x1, x2, x3], axis=1)

y2Pred, f2_pred = linearRegression(X, file['Number'].values)

x2New = np.power(xNew - 1971, 2)
x3_new = np.power(xNew - 1971, 3)
X = np.concatenate([xNew - 1971, x2New, x3_new], axis=1)
y2New = f2_pred(X)

plt.figure()
plt.plot(file['Year'], file['Number'], label='yTrue', color='blue')
plt.plot(file['Year'], yPred, label='yPred', color='red')
plt.plot(xNew, y_new, label='yPred to 2050', color='red', linestyle='--')
plt.plot(file['Year'], y2Pred, label='y2Pred', color='green')
plt.plot(xNew,y2New,label='polynomial to 2050',color='green',linestyle=':')
plt.legend()
plt.show()

#Question 2

from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat

file = pd.read_csv("C:/Users/Halil/Desktop/Pattern/Exercises/4.exercise/alabama.txt", sep=' ', header=None, names=['Year', 'Number'])

def triangular(x, a, b, c):
  y = np.where(x <= a, 0, x)
  y = np.where(np.logical_and(a < x, x <= b), (x - a) / (b - a), y)
  y = np.where(np.logical_and(b <= x, x < c), (c - x) / (c - b), y)
  y = np.where(x >= c, 0, y)
  return y


def song_fuzzyTime_prediction(x,y,n_sets=10,start=10000,stop=20000,overlap=1000,topK=1):
  x = x.ravel()
  y = y.ravel()
  xOrg = x
  yOrg = y
  assert len(y) == len(x), "Sample mismatch number"

  sets = np.linspace(start - overlap,stop + overlap,num=n_sets + 2,endpoint=True)[1:-1]
  fuzzy_sets = lambda idx, inputs: triangular(x=inputs, a=sets[idx] - overlap, b=sets[idx], c=sets[idx] + overlap)
  f = np.array([[fuzzy_sets(idx, yT) \
                 for idx in range(n_sets)] \
                for yT in y])
  fTop = np.argsort(f, axis=1)[..., ::-1]
  fTopk = fTop[:, :topK]

  rules = defaultdict(set)
  for start, end in zip(fTopk, fTopk[1:]):
    for s in start:
      for e in end:
        rules[s].add(e)

  yPred = []
  for xT, yT in zip(x, y):

    fT = np.array([fuzzy_sets(i, yT) for i in range(n_sets)])
    match_sets = np.argsort(fT)[::-1][:topK]

    compatible_rules = set()
    for f in match_sets:
      compatible_rules = compatible_rules | rules[f]

    x = sets[list(compatible_rules)]
    y = fT[list(compatible_rules)]
    coa = np.sum(x * y) / np.sum(y)
    yPred.append(coa)


  yT = yOrg[-1]
  xPrediction50 = []
  yPrediction50 = []
  for xT in range(1992, 1992 + 50):
    xPrediction50.append(xT)
    yPrediction50.append(yT)

    fT = np.array([fuzzy_sets(i, yT) for i in range(n_sets)])
    match_sets = np.argsort(fT)[::-1][:topK]
    compatible_rules = set()
    for f in match_sets:
      compatible_rules = compatible_rules | rules[f]
    x = sets[list(compatible_rules)]
    y = fT[list(compatible_rules)]
    coa = np.sum(x * y) / np.sum(y)

    yT = coa

  return yPred, [xPrediction50, yPrediction50]


y5Pred, (x50, y50) = song_fuzzyTime_prediction(file['Year'].values,file['Number'].values,n_sets=10,stop=30000,overlap=2000,topK=2)
plt.figure()
plt.plot(file['Year'], file['Number'], label='yTrue', color='red', linewidth=0.8)
plt.plot(file['Year'],y5Pred,label='y5Pred',color='blue',linestyle='--',linewidth=0.8)
plt.plot(x50,y50,label='next 50 years',color='green',linestyle='--',linewidth=0.8)

plt.legend()
plt.show()
