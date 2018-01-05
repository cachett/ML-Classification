#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import warnings
style.use('ggplot')

class Support_Vector_Machine:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        self.trained = False
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    #train
    def fit(self, data):
        self.trained = True
        self.data = data
        # { ||w|| : [w, b]}
        opt_dict = {}
        ###PLUS GROS PB DU PROGRAMME ON ESSAYE QUE 2 PENTES POUR L'HYPERPLAN!!! RAJOUTER DES ROTATIONS
        transforms = [[1,1], [-1,1], [1,-1], [-1,-1]]
        # rotMatrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
        #                  [np.sin(theta),  np.cos(theta)]])
        #
        # thetaStep = np.pi/10
        # transforms = [ (np.matrix(rotMatrix(theta)) * np.matrix([1,0]).T).T.tolist()[0]
        #                 for theta in np.arange(0,np.pi,thetaStep) ]
        all_data = []

        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      self.max_feature_value*0.001]

        #very expensive
        b_range_multiple = 2
        #we don't need to take as small of steps
        #with b as we do with w
        b_mutliple = 5

        lastest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([lastest_optimum, lastest_optimum])
            #we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_mutliple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    break
                        if found_option: #if every sample has fitted
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0 : # ca sert a rien d'aller plus loin car déja testé avec les transformations
                    optimized = True
                    print("optimized a setp")
                else:
                    w = w - step #w vector et step scalaire but ok with np
            norms = sorted([n for n in opt_dict]) # on trie les w qui vont bien par norme (clés)
            opt_choice = opt_dict[norms[0]] #on prend la plus petite
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            lastest_optimum = opt_choice[0][0] + step * 2


    def predict(self, features):
        #sign (x.w + b)
        if not self.trained:
            warnings.warn("trying to predict while SVM not trained")
        sign = np.sign(np.dot(np.array(features), self.w) + self.b)
        if sign != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker ='*', c=self.colors[sign])
        return sign

    def visualize(self):
        print(self.w)
        print(self.b)
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #v = w.x + b = hyperplane
        def hyperplane(x,w,b,v): #given an x value what is the y value for a given v, use v = w.x +b equation
            return (-w[0]*x-b+v)/w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #w.x +b = 1
        #positive support vector hyperplane
        positive1 = hyperplane(hyp_x_max, self.w, self.b, 1)
        positive2 = hyperplane(hyp_x_min, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [positive2, positive1], "k")

        #w.x +b = -1
        #positive support vector hyperplane
        negative1 = hyperplane(hyp_x_max, self.w, self.b, -1)
        negative2 = hyperplane(hyp_x_min, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [negative2, negative1], "k")

        #w.x +b = 0
        #positive support vector hyperplane
        db1 = hyperplane(hyp_x_max, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_min, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db2, db1], "y--")

        plt.show()

data_dict = {-1:np.array([[1,7], [2,8], [3,8]]), 1:np.array([[5,1], [6,-1], [7,3]])}
#data_dict = {1: np.array([(1,1),(2,1.5),(3,2)]), -1: np.array([(3,0),(4,0.5),(5,1)])}
#data_dict = {-1:np.array([[1,1], [2,1], [3,1],]),1:np.array([[1,4],[2,4],[3,4],])}
svm = Support_Vector_Machine()
svm.fit(data_dict)
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)
svm.visualize()
