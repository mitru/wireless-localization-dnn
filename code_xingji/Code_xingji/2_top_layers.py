#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:01:17 2018

@author: chenxingji
"""


import numpy as np

import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import statsmodels.api as sm

from Code.utilities import Plotting

# 这个脚本进行了顶层全连接层的训练和测试，这个过程只用了simon第一次收集的数据进行，
# 但是前面单独的wifi模型的训练用到了不只是simon的第一次采集的数据，同时也包括xingji采集的数据
# "./wifi_1/xing_output/output*.csv"存储的是wifi模型的输出结果
# “./wifi_1/simon_output/pre*.csv”存储的是sensor模型的输出结果
# =============================================================================
# 1. combine the two aligned output data from the wifi and the sensor models, 
#    as well as the ground of truth target.
# =============================================================================

#feature_DIM = 102
#feature_DIM = 135
feature_DIM = 164


# -----------------------------------------------------------------------------------
def combine_two_output(suffix):
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
        
    wifi_path = "./wifi_"+ ss +"/xing_output(example)/output"+str(suffix)+".csv"
    sensors_path = "./wifi_"+ ss +"/simon_output/pre"+str(suffix)+".csv"
    
    tar_pre1 = np.array(pd.read_csv(wifi_path, header=None))
    pre2 = np.array(pd.read_csv(sensors_path, index_col=0))
    
    tar_pre1_pre2 = np.hstack((tar_pre1, pre2)) 
    # tar is the target, pre1 is wifi model's prediction, pre2 is sensor model's prediction
    
    return tar_pre1_pre2

# read the csv file "simon_wifi_*.csv" into dataframe and shuffle it
def set_shuffle(dataset):
    
    np.random.shuffle(dataset)
    # get concatenated 4 input
    X = dataset[:, 2:]
    # get formatted and normalised target outputs(x,y)- 2 values
    Y = dataset[:, 0:2]
    
    return X, Y
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def select_train_val_test(mode):
    # training set
    if mode == 'train':
        dataset = combine_two_output(1) # 1
        print(np.shape(dataset))
        for i in range(7):  # from 2 to 8
            i = i+2
            cur = combine_two_output(suffix = i)
            print(np.shape(cur))
            dataset = np.vstack((dataset, cur))
        print("training set shape: {}".format(np.shape(dataset)))
    # validation set
    elif mode == 'val':
        dataset = combine_two_output(9) # 9 
        print(np.shape(dataset))
        for i in range(4):  # from 10 to 13
            i = i+10
            cur = combine_two_output(suffix = i)
            print(np.shape(cur))
            dataset = np.vstack((dataset, cur))
        print("validation set shape: {}".format(np.shape(dataset)))
    # test set
    elif mode == 'test':
        dataset = combine_two_output(14)
        print("test set shape: {}".format(np.shape(dataset)))
    
    x, y = set_shuffle(dataset)
    return x, y
# -----------------------------------------------------------------------------------










# =============================================================================
# 2. train the top 2 fully connected layers based on the data from above
# =============================================================================

# --222------------------------------------------------------------------------------
# The following 3 functions is used to get the test set cdf plot
north_west = (55.945139, -3.18781)  # A
south_east = (55.944600, -3.186537)  # B

def get_distance(lnglat1, lnglat2):
    '''
    get the distance in meters of two location
    :param lnglat1:
    :param lnglat2:
    :return: distance in meters
    '''
    rr = 6381 * 1000

    lng1 = lnglat1[0]
    lat1 = lnglat1[1]

    lng2 = lnglat2[0]
    lat2 = lnglat2[1]

    lng_distance = math.radians(lng2 - lng1)
    lat_distance = math.radians(lat2 - lat1)

    a = pow(math.sin(lat_distance / 2), 2) \
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) \
        * pow(math.sin(lng_distance / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = rr * c
    return distance   
     
def transfer_error_in_meters(data):
    
    lnglat = np.zeros(np.shape(data))
    # lng
    lnglat[:, 0] = north_west[1] + ((data[:, 0] + 1) * (south_east[1] - north_west[1]) / 2)
    lnglat[:, 2] = north_west[1] + ((data[:, 2] + 1) * (south_east[1] - north_west[1]) / 2)
    # lat
    lnglat[:, 1] = north_west[0] - ((data[:, 1] + 1) * (north_west[0] - south_east[0]) / 2)
    lnglat[:, 3] = north_west[0] - ((data[:, 3] + 1) * (north_west[0] - south_east[0]) / 2)

    errors = np.zeros((np.shape(lnglat)[0], 1))

    for item, i in zip(lnglat, range(np.shape(errors)[0])):
        errors[i, 0] = get_distance([item[0], item[1]], [item[2], item[3]])

    return errors  

# correct implementation version
def cdf_plot(data, name, number):
    ecdf = sm.distributions.ECDF(data)
    length = len(data)

    # x = np.linspace(min(data), max(data), number)
    lower_bound = ecdf.x[int(length*0.1)]
    upper_bound = ecdf.x[int(length*0.9)]
    x = np.linspace(lower_bound, upper_bound, number)
    y = ecdf(x)

    print("x ~ [{}, {}]".format(round(ecdf.x[0], 3), round(ecdf.x[length], 3)))
    print("mid_x={}\t\t{}\n".format(round(ecdf.x[int(length/2)], 3), name))

    # plt.step(x, y, label=name)
    plt.plot(x, y, label=name)  
# -----------------------------------------------------------------------------------

# -111----------------------------------------------------------------------------------
#    根据interim_output文件夹中的test_output{*}.txt生成e{*}.txt，并用它来绘制cdf
def error_in_meter_plotcdf():
    suffix = "(top)"
        
    fn = "./interim_output/test_output"+suffix+".txt"
    target_and_neural_out = np.loadtxt(fn, delimiter=',')

    error_in_meters = transfer_error_in_meters(target_and_neural_out)
    with open("./interim_output/e"+suffix+".txt", "w") as f:
        np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    fig = plt.figure()
    data = np.loadtxt("./interim_output/e"+suffix+".txt")
    cdf_plot(data, "Top model", 100)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("./graph_output/CDF"+suffix+".png")
    print("\n")
# -----------------------------------------------------------------------------------






# some visualization functions
# -111------------------------------------------------------------------------------
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
    ax.set_yticks(my_y_ticks, minor=False)

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

    ax.set_title("Errors of regression{}".format(suffix))
    ax.legend()
    plt.show()

    # save error line fig
    fig.savefig("./graph_output/errors_visualization"+str(suffix)+".png")  # regression [200,200,200]

# save neural output to file "./interim_output/test_output{*}.txt"
def save_results(Y_test, Y_pre, suffix):
    Y_test = np.array(Y_test)
       
    txt_filename = "./interim_output/test_output"+str(suffix)+".txt"
    write_text = np.hstack((Y_test, Y_pre))

    # write the target output and predicted output into "test_output_2.txt"
    with open(txt_filename, "wb") as f:
        np.savetxt(f, write_text, delimiter=",", newline='\n')
# -----------------------------------------------------------------------------------


def top_main(x_train, y_train, x_val, y_val, x_test, y_test):
    
    ss = "(top)"
    
    INPUT_DIM = 4
    OUTPUT_DIM = 2
    hidden_num = [100,100]
    
    # ---Constructing neural network---
    model = Sequential()
    model.add(Dense(hidden_num[0], activation="relu", input_dim=INPUT_DIM))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_num[1], activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(activation="tanh", output_dim=OUTPUT_DIM))
    sgd = SGD(lr=0.01, decay=1e-8, momentum=0.90, nesterov=True)
    # 0.001搭配1e-6的cdf可以达到0.6，训练曲线平滑下降，且一直在降，未降到底
    # momentum低于0.9的时候好像xy都训练很不到位，预测点都集中在原点区域
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
    # the metric function is similar to a loss function, except that the results
    # from evaluating a metric are not used when training the model.

    # ---Training a neural network--- 

    # fit model
#    model.fit(x_train, y_train, epochs=200, batch_size=8, callbacks=[checkpoint], validation_data=[x_test, y_test], verbose=1)
    model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=[x_test, y_test], verbose=1)
    model.save("./models/top_model_6.h5")
    
    
    # ---Ploting the acc & loss curve throughout the training process---
    Plotting.plot_train_val(model_history=model.history.history, is_classification=False, suffix=ss)

    # ---Getting test set performance---
    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
    
    
    # generate the error line png into 'graph_output' directory
    visualization(Y_test=y_test, Y_pre=y_pre, suffix=ss)
    # save the tar&pre into "./interim_output/test_output{*}.txt"
    save_results(Y_test=y_test, Y_pre=y_pre, suffix=ss)  # baseline [200,200,200]

    print("End of training.")
    
    
    
# =============================================================================
# Main Function
# =============================================================================

if __name__ == "__main__":
    
    
#   finish 1st step: get train, val, test set
    x_train, y_train = select_train_val_test(mode='train')
    x_val, y_val = select_train_val_test(mode='val')
    x_test, y_test = select_train_val_test(mode='test')
    
    # finish 2nd step: 
    top_main(x_train, y_train, x_val, y_val, x_test, y_test)
    error_in_meter_plotcdf()    # 指定验证集，验证已保存的模型
