#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
style.use('ggplot')


X = np.array([[1,2],[1.5,1.8],[5,8],[1,0.6],[8,8],[9,11]])
colors = 100*["g","r","c.","b.","k.","o."]


class KMeans:
    def __init__(self, k=2, tol=0.0001, max_iter=300):#tolerence reprensent in % the movement of centroids in 1 iteration
        self.k = k
        self.tolerence = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for k in range(self.k):
                self.classifications[k] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroid = dict(self.centroids) #We do that for object pointer python3

            for classification in self.classifications: #redefinied new centroids from classification
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                orginal_centroid = prev_centroid[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - orginal_centroid) / orginal_centroid * 100) > self.tolerence:
                    optimized = False

            if optimized: # we stop if the centroids don't move more than tolerence
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = KMeans()
clf.fit(X)
to_predict = [6,4]
predicted = clf.predict(to_predict)
plt.scatter(to_predict[0], to_predict[1], color=colors[predicted], s=150, marker='X', linewidths=5)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o',color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='x', color=color, s=150, linewidths=5)

plt.show()
