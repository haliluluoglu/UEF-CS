##Exercise1
#Question2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import seaborn as sns

def Negative_Selection_Algorithm(data, radius=0.2, nDetectors=10):
    detectors=[]
    minValue=np.min(data, axis=0, keepdims=True)
    maxValue=np.max(data, axis=0, keepdims=True)
    data=(data-minValue)/(maxValue-minValue)

    while True:
        center=np.random.rand(2)[np.newaxis, :]
        distances=np.linalg.norm(data-center, axis=1)
        if np.all(distances > radius):
            detectors.append(center)
            if len(detectors) > nDetectors:
                break
    print(detectors)
    return detectors


X,y = load_iris(return_X_y=True)
print(X.shape, y.shape)
print(y)
X=PCA(n_components=2).fit_transform(X)


detectors = Negative_Selection_Algorithm(X)

plt.figure(figsize=(10,5))
sns.scatterplot(x=X[:,0], y=X[:, 1],hue=y)
plt.show()
