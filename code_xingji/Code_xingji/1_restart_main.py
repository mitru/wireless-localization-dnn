#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 2018, 16:15:20

@author: chenxingji

"""

# =============================================================================
# This file 'restart_main.py' is using data generated from 'combineData.py', 
# that is 'both_wifi(train).csv', 'both_wifi(test).csv' to train the network from scratch.
# save the new model as './models/restart_model.h5'

# 'both_wifi(train).csv' combined both simon and xingji's training set
# 'both_wifi(test).csv' combined both simon and xingji's test set

# 第一种（original）和第四种（restart）模型都用此脚本训练及产生数据
# =============================================================================

import numpy as np
import math

np.random.seed(100)
import tensorflow as tf

tf.set_random_seed(100)

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import statsmodels.api as sm

from Code.utilities import Plotting


# global varibles
#feature_DIM = 102  # xingji
#feature_DIM = 135  # simon
feature_DIM = 164  # both
 
# --222------------------------------------------------------------------------------
# read the csv file "simon_wifi_*.csv" into dataframe and shuffle it
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
def error_in_meter_plotcdf(is_original):
    if is_original:
        suffix = "(original)"
    else:
        suffix = "(restart)"
        
    fn = "./interim_output/test_output"+suffix+".txt"
    target_and_neural_out = np.loadtxt(fn, delimiter=',')

    error_in_meters = transfer_error_in_meters(target_and_neural_out)
    with open("./interim_output/e"+suffix+".txt", "w") as f:
        np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    fig = plt.figure()
    data = np.loadtxt("./interim_output/e"+suffix+".txt")
    cdf_plot(data, "reg model"+suffix, 100)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("./graph_output/CDF"+suffix+".png")
    print("\n")
# -----------------------------------------------------------------------------------


# -111-------------------------------------------------------------------------------
def restart_main(is_original):
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
    
    
    # Read training/test data from csv file
    if is_original:                     # original mode(training model using xingji's data)
        train_file = "./Data/train_test_"+ ss +"/simon_wifi(train).csv"     # xingji's training set
        test_file = "./Data//train_test_"+ ss +"/simon_wifi(test).csv"      # xingji's test set
    else:                               # restart mode(training model from scratch using both data)
        train_file = "./Data/train_test_"+ ss +"/both3_wifi(train).csv"     # both training set
        test_file = "./Data/train_test_"+ ss +"/both3_wifi(test).csv"       # both/simon/xingji test set
        
    x_train, y_train = data_shuffle(train_file)
    x_test, y_test = data_shuffle(test_file)
    
    INPUT_DIM = feature_DIM
    OUTPUT_DIM = 2
    hidden_num = [200,200,200]
    
    # ---Constructing neural network---
    model = Sequential()
    model.add(Dense(hidden_num[0], activation="relu", input_dim=INPUT_DIM))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_num[1], activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_num[2], activation="relu"))
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

    if is_original:
        s1 = "(original)"
    else:
        s1 = "(restart)"
        
    # fit model
    model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=[x_test, y_test], verbose=1)
    model.save("./models/"+str(s1[1:-1])+"_model_6.h5")
    
    # ---Ploting the acc & loss curve throughout the training process---
    Plotting.plot_train_val(model_history=model.history.history, is_classification=False, suffix=s1)

    # ---Getting test set performance---
    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
        
    # generate the error line png into 'graph_output' directory
    visualization(Y_test=y_test, Y_pre=y_pre, suffix=s1)
    # save the tar&pre into "./interim_output/test_output{*}.txt"
    save_results(Y_test=y_test, Y_pre=y_pre, suffix=s1)  # baseline [200,200,200]

    print("End of training.")
# --------------------------------------------------------------------------------



def ttest(is_original):

    if is_original:
        ss = "(original)"
    else:
        ss = "(restart)"
        
    if feature_DIM == 102:
        sss = "1"
    elif feature_DIM == 135:
        sss = "2"
    elif feature_DIM == 164:
        sss = "3"
        
    test_file = "./Data/train_test_"+ sss +"/both_wifi(test).csv"       # both test set
    x_test, y_test = data_shuffle(test_file)
    
    # ---load model---
    model = load_model("./models/"+str(ss[1:-1])+"_model.h5")
    model.summary()


    # ---Getting test set performance---
    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
    
    
    # generate the error line png into 'graph_output' directory
    visualization(Y_test=y_test, Y_pre=y_pre, suffix=ss)
    # save the tar&pre into "./interim_output/test_output{*}.txt"
    save_results(Y_test=y_test, Y_pre=y_pre, suffix=ss)  # baseline [200,200,200]

    

if __name__ == "__main__":
#    执行restart_main会在graph_output文件夹中生成一张图，并在interim_output文件夹中生成两个.txt文件
#    restart_main相当于将集成了训练，测试，绘制训练曲线，绘制error line，存2个txt，保存测试集输出结果和目标
#    ttest用于指定一个test集，加载一个已保存的模型，对该模型在test set上的表现进行评分,绘制error line，存2个txt，保存测试集输出结果和目标
#    error_in_meter_plotcdf从之前两个函数生成的测试集的输出文件txt，得到e.txt，用来绘制cdf    

#    所以一般一趟要嘛执行restart_main和error_in_meter_plotcdf的组合，要嘛执行ttest和error_in_meter_plotcdf的组合
    flag = False # True: original mode, train the original model using only xingji's data; 
                # False: restart mode, training by both data from scratch;
                
    restart_main(is_original=flag)
#    ttest(is_original=flag)
    error_in_meter_plotcdf(is_original=flag)    # 指定验证集，验证已保存的模型

