#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')

X, y = make_blobs(n_samples=50, centers = 3, n_features=2)
# X = np.array([[1,2],[1.5,1.8],[5,8],[1,0.6],[8,8],[9,11],[8,2],[10,2],[9,3]])
colors = 100*["g","r","c","b","k","o"]
# plt.scatter(X[:,0], X[:,1], color="b", s=150, linewidths=5, marker="o")
# plt.show()



class MeanShift:
    def __init__(self, radius=None, radius_norm_step=80):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        weights = [i**2 for i in range(self.radius_norm_step)][::-1]


        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for feature_set in data:
                    # if np.linalg.norm(feature_set-centroid) < self.radius:
                    #     in_bandwidth.append(feature_set)

                    #Here we are weight sample by distance so we don't have to provide radius
                    distance = np.linalg.norm(feature_set-centroid)
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    to_add = (weights[weight_index])*[feature_set]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))#Tuple to be able to use set to get unique value, np.unique
                #return unique "array" and not value ...

            uniques = sorted(list(set(new_centroids)))

            #Here we're popping the centroids which are close enough
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= 0.01: #tolerence
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break
            if optimized:
                break

        #Here we put informations in our object
        self.centroids = new_centroids
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for feature_set in data:
            distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in range(len(self.centroids))]
            classification = distances.index(min(distances))
            self.classifications[classification].append(feature_set)


    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in range(len(self.centroids))]
        classification = distances.index(min(distances))
        return classification

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='o', color = color, s=150)

for c in range(len(centroids)):
    plt.scatter(centroids[c][0], centroids[c][1], color="k", s=150, linewidths=5, marker="x")
plt.show()
