#!/usr/bin/env python3

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import random
# xs = np.array([1,2,3,4,5,6])
# ys = np.array([2,4,5,5,7,4])

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys= []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)*mean(xs) - mean(xs*xs))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line)**2)

def coefficient_of_determination(ys_orign, ys_line):
    y_mean_line = [mean(ys_orign) for _ in ys_orign]
    squared_error_regr = squared_error(ys_orign, ys_line)
    squared_error_mean_line = squared_error(ys_orign, y_mean_line)
    return 1 - squared_error_regr/squared_error_mean_line

xs, ys = create_dataset(400, 300, 2, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x + b) for x in xs]
coefficient_of_determination = coefficient_of_determination(ys, regression_line)
print(coefficient_of_determination)

predict_x = 8
predict_y = m*predict_x + b

plt.scatter(predict_x, predict_y, s=100, color="G")
plt.plot(xs, regression_line)
plt.scatter(xs, ys)
plt.show()
