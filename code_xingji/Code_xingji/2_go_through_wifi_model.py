#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:39:16 2018

@author: chenxingji

"""
# 将0.1s间隔的wifi数据输入wifi网络，将wifi网络的输出和ground-truth记下来，用于训练和测试最顶层的两层模型
# =============================================================================
# this file is used to generate the standard alighed data that can be feed to the top 2 fully connected layers
# from "sampled{*}.csv" into "output{*}.csv" [x-tar, y-tar, x-pre, y-pre]
# =============================================================================

import re
import numpy as np
import pandas as pd
from keras.models import load_model

import matplotlib.pyplot as plt


#feature_DIM = 102
#feature_DIM = 135
feature_DIM = 164


# ---333--------------------------------------------------------------------------------
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
def save_results(target, predict, suffix):
    target = np.array(target)
    write_text = np.hstack((target, predict))
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
       
    txt_filename = "./wifi_"+ ss +"/xing_output(example)/output"+str(suffix)+".csv"
    # write the target output and predicted output into "output{*}.txt"
    with open(txt_filename, "wb") as f:
        np.savetxt(f, write_text, delimiter=",", newline='\n')
# -----------------------------------------------------------------------------------










# ---222------------------------------------------------------------------------------
# read the csv file "simon_wifi_*.csv" into dataframe and shuffle it
def data_split(name):
    
    dataset = pd.read_csv(name)

    # col_list is the APs columns name, that is: ap0,ap1,... ap102 
    col_list = []
    for i in range(feature_DIM):    # 102
        col_name = 'ap' + str(i)
        col_list.append(col_name)
        
    # get formatted nueral network inputs(standardised wifi signal)- 102 values
    X = dataset[col_list]
  
    # get formatted and normalised target outputs(x,y)- 2 values
    Y = dataset[['x-lng', 'y-lat']]

    return X, Y
# --------------------------------------------------------------------------------









# -111-------------------------------------------------------------------------------
def get_top_data(path):
    fname = path.split("/")[-1]
    ss = re.search(r'\d+', fname).group(0)
    
    x_test, y_test = data_split(path)
    
    # ---load model---
    model = load_model("./models/restart_model_6.h5")
    model.summary()

    # ---Getting test set performance---
    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
    
    
    # generate the error line png into 'graph_output' directory
    visualization(Y_test=y_test, Y_pre=y_pre, suffix=ss)
    
    # save the tar&pre into "./wifi_1/output/output{*}.txt"
    save_results(target=y_test, predict=y_pre, suffix=ss)  # baseline [200,200,200]
# --------------------------------------------------------------------------------

    

    
if __name__ == "__main__":
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
     # 将sampled*.csv的每个输入wifi喂给单独的wifi模型，并记录下单独wifi模型的输出  
    for i in range(14):
        i = i+1
        fpath = "./wifi_"+ ss +"/sampled/sampled"+str(i)+".csv"
        get_top_data(fpath)
    
#    fpath = "./wifi_1/sample/sampled1.csv"
#    get_top_data(fpath)
    
    
