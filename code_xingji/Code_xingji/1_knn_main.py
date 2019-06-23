#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:17:31 2019

@author: chenxingji
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ShuffleSplit, KFold, GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt


feature_DIM = 102



# --222------------------------------------------------------------------------------
# error line plot into file "/graph_output/errors_visualization{*}.png"
def visualization(Y_test, Y_pre, suffix):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel("x-longitude")
    ax.set_ylabel("y-latitude")

    # 设置x,y主坐标轴
    my_x_ticks = np.arange(-40, 40, 10)
    my_y_ticks = np.arange(-30, 30, 10)
    ax.set_xticks(my_x_ticks, minor=False)

    # 设置x,y次坐标轴
    my_x_ticks = np.arange(-40, 40, 2)
    my_y_ticks = np.arange(-30, 30, 2)
    ax.set_xticks(my_x_ticks, minor=True)
    ax.set_yticks(my_y_ticks, minor=True)

    ax.set_xlim((-40, 40))
    ax.set_ylim((-30, 30))

    Y_test = np.array(Y_test)
    for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
        plt.plot([int(pred[0] * 40), int(target[0] * 40)], [int(pred[1] * 30), int(target[1] * 30)], color='r',
                 linewidth=0.5, label='error line' if i == 0 else "")
        plt.scatter(int(pred[0] * 40), int(pred[1] * 30), label='prediction' if i == 0 else "", color='b', marker='.')
        plt.scatter(int(target[0] * 40), int(target[1] * 30), label='target' if i == 0 else "", color='c', marker='.')

    ax.set_title("Errors of regression-{}".format(suffix))
    ax.legend()
    plt.show()

    # save error line fig
    fig.savefig("./graph_output/errors_visualization_"+str(suffix)+".png")  # regression [200,200,200]


# save neural output to file "./interim_output/test_output{*}.txt"
def save_results(Y_test, Y_pre, suffix):
    Y_test = np.array(Y_test)
    
    txt_filename = "./interim_output/test_output_"+str(suffix)+".txt"
    write_text = np.hstack((Y_test, Y_pre))

    # write the target output and predicted output into "test_output_2.txt"
    with open(txt_filename, "wb") as f:
        np.savetxt(f, write_text, delimiter=",", newline='\n')
# -----------------------------------------------------------------------------------



# --222------------------------------------------------------------------------------
# read the csv file "simon_wifi(*).csv" into dataframe and shuffle it
def data_shuffle(name):
    
    dataset = pd.read_csv(name)
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

    # col_list is the APs columns name, that is: ap0,ap1,... ap102 
    col_list = []
    for i in range(feature_DIM):    # 102
        col_name = 'ap' + str(i)
        col_list.append(col_name)
        
    # get formatted nueral network inputs(standardised wifi signal)- 102 values
    X = shuffled_dataset[col_list]
    # get formatted and normalised target outputs(x,y)- 2 values
    Y = shuffled_dataset[['x-lng', 'y-lat']]
    
    return X, Y
# -----------------------------------------------------------------------------------



from sklearn.metrics import mean_squared_error
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    score = mean_squared_error(y_true, y_predict)
   
    return score


# 用ShuffleSplit 或者 KFold方法 找KNN算法的最佳K值
def select_best_k(X, y, mode):
    # Create cross-validation sets from the training data
    if mode == 'Shuffle':
        cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    elif mode == 'KFold':
        cv_sets = KFold(n_splits=10)
    
    regressor = KNeighborsRegressor()
    params = {'n_neighbors':range(3,10)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
  
    

train_file = "./Data/train_test_1/both3_wifi(train).csv"     # both training set
test_file = "./Data/train_test_1/both3_wifi(test).csv"
x_train, y_train = data_shuffle(train_file)
x_test, y_test = data_shuffle(test_file)

best1 = select_best_k(x_train, y_train, 'Shuffle')
#best2 = select_best_k(x_train, y_train, 'KFold')
print(best1)
#print(best2)

pre_test = best1.predict(x_test)
pre_train = best1.predict(x_train)


print(performance_metric(y_test, best1.predict(x_test)))
# generate the error line png into 'graph_output' directory
visualization(Y_test=y_test, Y_pre=pre_test, suffix='knn')
# save the tar&pre into "./interim_output/test_output{*}.txt"
save_results(Y_test=y_test, Y_pre=pre_test, suffix='knn')  # baseline [200,200,200]



