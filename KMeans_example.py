#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
style.use('ggplot')


X = np.array([[1,2],[1.5,1.8],[5,8],[1,0.6],[8,8],[9,11]])
# plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize =20)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=150, linewidths=5)
plt.show()
