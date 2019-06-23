# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Chen Xingji 
Date: 2018/11/12
"""

# 本文件将xingji和simon收集的数据合并，依次产生wifi模型的训练数据和测试数据,得到 both(train).csv和 both(test).csv

# combineData.py可分为三部分，第一部分是从simon的所有数据中产生2个总的文件: "./Data/train_test/simon_wifi(train）.csv", "./Data/train_test/simon_wifi(test).csv"
# 第二部分是从xing的数据’out_in_overall.csv‘中产生2个总的文件："./Data/train_test/xing_wifi(train/test).csv"
# 第三部分是合并两人的训练数据，合并两人的测试数据，得到两个总文件："./Data/train_test/both_wifi(train/test).csv"
# 第三部分也可合并3个数据集（通过运行combine_three_dataset函数合并simon+ xing+ example），得到两个总文件："./Data/train_test/both3_wifi(train/test).csv"
#
# =============================================================================
#  
# This file is Combining Simon's data and Xingji's data for training the network from scratch.

# this generate a csv files called "./Data/both_wifi(train).csv" containing [t, x-lng, y-lat, ap0, ..., ap101] fields
# and also another csv file called "./Data/both_wifi(test).csv" containing [t, x-lng, y-lat, ap0, ..., ap101] fields

# 本脚本中，对simon的wifi数据和location数据进行的归一化处理与之前旧模型用的数据的归一化处理是一致的。
# 
# =============================================================================

import os
import collections 
import re
import xml.dom.minidom
import h5py

import numpy as np
import pandas as pd
from scipy import interpolate
#from collections import Counter

# global varibles
#feature_DIM = 102  # xingji
#feature_DIM = 135  # simon
#feature_DIM = 164  # both


# # ---333----------------------------------------------------------------------
# Read the 102 distinct APs from Simon's data to 'WIFI_DICT'
# "./Data/wifi_id1.txt": distinct APs extracted from Xingji's data   
# "./Data/wifi_id2.txt": distinct APs extracted from Simon's data  **applied**

def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict
# ----------------------------------------------------------------------------------------

wifi_filename = "./Data/wifi_id3.txt"

WIFI_DICT = read_ap_to_dict(wifi_filename)

feature_DIM = len(WIFI_DICT)




# # ---333----------------------------------------------------------------------
# Read the 164 distinct APs from the uninon set of Xingji's & Simon's data to 'WIFI_DICT'
# "./Data/wifi_id1.txt": distinct APs extracted from Xingji's data 
# "./Data/wifi_id2.txt": distinct APs extracted from Simon's data 
# ** BOTH APs has been APPLIED **

#def read_ap_to_dict(filename):
#    ap_dict = {}
#    with open(filename) as file:
#        for line in file:
#            elements = re.split(r'[\s]', line.strip())
#            ap_dict[elements[0]] = int(elements[1])
#    return ap_dict
#
#wifi_filename1 = "./Data/wifi_id1.txt"
#wifi_filename2 = "./Data/wifi_id2.txt"
#w1 = read_ap_to_dict(wifi_filename1)
#w2 = read_ap_to_dict(wifi_filename2)
#
#z = dict(Counter(w1)+Counter(w2))
#sorted_z = sorted(zip(z.values(), z.keys()))[::-1]
#
#WIFI_DICT = collections.OrderedDict()
#i = 0
#for v,k in sorted_z:
#    i += 1
#    WIFI_DICT[k] = (str(v), str(i))
#feature_DIM = len(WIFI_DICT)






# ---333-------------------------------------------------------------------------------------
# Normalise each APs strength to [0,1]
def normalize_wifi_inputs(wr_inputs):

    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    max = -40
    min = -100
    wr_inputs = (wr_inputs - min) / (max - min)

    return wr_inputs

# convert the original lat&lng to [-1,1] 
def latlng_to_cor(outputs):
    north_west = (55.945139, -3.18781)  # A
    south_east = (55.944600, -3.186537)  # B
    
    # lat-y
    max0 = north_west[0]
    min0 = south_east[0]
    outputs[:, 0] = 2 * (outputs[:, 0] - min0) / (max0 - min0) - 1

    # lng-x
    max1 = south_east[1]
    min1 = north_west[1]
    outputs[:, 1] = 2 * (outputs[:, 1] - min1) / (max1 - min1) - 1

    # now the outputs[lat, lng], that is [y, x]
    # we would like to reverse the order of the 2 columns, and become [x,y] as follow:
    outputs[:, [0, 1]] = outputs[:, [1, 0]]

    return outputs

# Generate the interpolated location according to the input time list<t_list>
def get_label_list(prefix, t_list):
    csvFile = open("./Archive/"+prefix+".csv", "r")
    reader = pd.read_csv(csvFile)
    print(reader)
    del reader['lat']
    del reader['Lng']
    del reader['x']
    del reader['y']

    time_serial = reader["Time"]
    
    t1 = time_serial[0]
    t2 = time_serial[4]
#    xnew = np.linspace(t1, t2, 1000*(t2-t1)+1)

    f_lat = interpolate.interp1d(reader['Time'], reader['lat1'], kind='linear')
    f_lng = interpolate.interp1d(reader['Time'], reader['Lng1'], kind='linear')
    c123 = []
    
    for _t in t_list:
        time_x = round(t1 + (0.001 * _t), 3)
        if time_x > t2:
            break
        else:
            interpolated_lat = f_lat(time_x).ravel()[0]
            interpolated_lng = f_lng(time_x).ravel()[0]
            t_lat_lng = [time_x, interpolated_lat, interpolated_lng]
            c123.append(t_lat_lng)

    c123 = np.array(c123)
    c123[:, 1:] = latlng_to_cor(c123[:, 1:])
    csvFile.close()
    
    return c123
# ----------------------------------------------------------------------------------------






# 根据wifi scan产生样本，一个wifi scan是一个样本
# --222--------------------------------------------------------------------------------------
# scan_wifi_and_write utilised "get_label_list" and "normalize_wifi_inputs"
def scan_wifi_and_write(file_name):
    # 记录一个文件中的所有记录的wifi
    _wifi_list = list()
    _t_list = list()

    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement

    wr_list = root.getElementsByTagName('wr')
    a_list = root.getElementsByTagName('a')
    m_list = root.getElementsByTagName('m')
    
    
    # record the starting time of this file    
    start = min(a_list[0].getAttribute("t"), m_list[0].getAttribute("t"))

    for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
        sum = 0
        t = item.getAttribute("t")
        t = int(t) - int(start)
        # wifi_record用于记录此时间步的wr向量
        wifi_record = []  # wifi_record is a list, which contains len(wifi_dict) elements
        for _ in range(len(WIFI_DICT.keys())):
            wifi_record.append(0)
        
        # 为wifi向量的每一维度赋值        
        for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
            if j % 2:
                ap_id = item.childNodes[j].getAttribute("b")
                ap_v = item.childNodes[j].getAttribute("s")
                if ap_id in WIFI_DICT.keys():
                    index = int(WIFI_DICT[ap_id][1]) - 1
                    wifi_record[index] = int(ap_v)
                else:
                    sum += 1
                    # print("{} not in wifi_dict1.".format(ap_id))
                    
        # wifi_list表示一个xml文件中所有的wifi记录
        _wifi_list.append(wifi_record)
        _t_list.append(t)
#        print("time:"+ str(i)+" sum:"+str(sum))
        
    # 处理完整个xml文件了，所需信息都记在wifi_list里了
    prefix = file_name.split("/")[-1].split(".")[0]

    wifi_shape = (len(_wifi_list), len(WIFI_DICT.keys()))
    wifi_records = np.reshape(_wifi_list, wifi_shape)

    # 时间t和位置标签xy
    c_123 = get_label_list(prefix, _t_list)

    # wifi数组
    wifi_array = normalize_wifi_inputs(wifi_records)
    num = c_123.shape[0]
    c_4 = wifi_array[:num, :]     # 注意这里用的是wifi_record的长度，而不是wifi_records
    
    if c_123.shape[0] == c_4.shape[0]:
        file_content = np.concatenate((c_123, c_4), axis=1)
    else:
        print("size not consistent")

    # write <t,x,y,wifi>  to 'tXY_f*.csv' file
    file_df = pd.DataFrame(file_content, columns=['t', 'x-lng', 'y-lat']+['ap'+str(i) for i in range(len(wifi_record))])
#    with open('tYX_f'+prefix+'.csv', 'w') as writeFile:
#        file_df.to_csv(writeFile, index=False, sep=',')

    print("current sample size: {}\n".format(np.shape(file_content)))
    
    return file_df
# ----------------------------------------------------------------------------------------








# 获取 “./Data/xing_wifi(train).csv'和 ’./Data/xing_wifi(t).csv‘
# -111-------------------------------------------------------------------------------------
# traverse_all utilised the function 'scan_wifi_and_write()' 
def traverse_all(path):
    '''
    traverse all simon's xml files and get all wifi records
    convert them into standard input and write those 't' & 'x,y' & 'wifi' data into file
    
    '''
    dirs = os.listdir(path)
    total_df = pd.DataFrame(columns = ['t', 'x-lng', 'y-lat']+['ap'+str(i) for i in range(len(WIFI_DICT))])

#   2. Extract Simon's training set
    for dd in dirs:
        if dd.endswith('.xml'):
            
            ll = dd.split('.')
            if (0 < int(ll[0]) < 9):
#            if (8 < int(ll[0]) < 15):
                f_id = os.path.join(path, dd)
                print("processing... ", f_id)
                cur_df = scan_wifi_and_write(f_id)
                total_df = pd.concat([total_df, cur_df])
    
    with open('./Data/train_test/simon_wifi(train).csv', 'w') as file:
        total_df.to_csv(file, sep=',')
# ----------------------------------------------------------------------------------------






# 获取 “./Data/train_test/xing_wifi(train).csv'和 ’./Data/train_test/xing_wifi(test).csv‘
# -111-------------------------------------------------------------------------------------
def extract_saved_xing():
    dataset = h5py.File("out_in_overall_135.h5", "r")

    # read structured data from "dataset"(h5py file) into "X_Y"(numpy array)
    for name, i in zip(dataset.keys(), range(len(dataset.keys()))):
        print(np.shape(dataset[name]))
        if i == 0:
            X_Y = dataset[name]
        else:
            X_Y = np.vstack((X_Y, dataset[name]))
        print("--------")
    dataset.close()
     
    np.random.shuffle(X_Y)
    SAMPLE_NUM = np.shape(X_Y)[0]
    

    # get formatted nueral network inputs(standardised wifi signal)- 102 values
    X = X_Y[:, 5:]
    X = normalize_wifi_inputs(X[:, 600:])  # get normalized wifi inputs(The parameters passed in by this function are including the
    #  102 dimensions wifi features.)

    # get formatted and normalised target outputs(x,y)- 2 values
    Y = X_Y[:, :2]
    Y = latlng_to_cor(Y)

    chunk_size = int(0.2 * SAMPLE_NUM)
    Y_test = Y[0:chunk_size, :]
    X_test = X[0:chunk_size, :]

    Y_train = Y[chunk_size:, :]
    X_train = X[chunk_size:, :]

#    np.savetxt("./Data/xing_wifi(train).csv", np.hstack((Y_train, X_train)), delimiter=",")
#    np.savetxt("./Data/xing_wifi(test).csv", np.hstack((Y_test, X_test)), delimiter=",")
    
    train_data = np.hstack((Y_train, X_train))
    test_data = np.hstack((Y_test, X_test))
    print("train:{}".format(np.shape(train_data)))
    print("test:{}".format(np.shape(test_data)))
    
    train_df = pd.DataFrame(train_data, columns=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    test_df = pd.DataFrame(test_data, columns=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    
    with open('./Data/train_test_2/xing_wifi(train).csv', 'w') as file:
        train_df.to_csv(file, sep=',')
        
    with open('./Data/train_test_2/xing_wifi(test).csv', 'w') as file:
        test_df.to_csv(file, sep=',')   
# ----------------------------------------------------------------------------------------





# -111-------------------------------------------------------------------------------------
def combine_two_dataset():
    # combine 2 training set
    csvFile1 = open("./Data/train_test_2/simon_wifi(train).csv", "r")
    csvFile2 = open("./Data/train_test_2/xing_wifi(train).csv", "r")
    
    reader1 = pd.read_csv(csvFile1, usecols=['t', 'x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
#    print(reader1)
    del reader1['t']
    reader2 = pd.read_csv(csvFile2, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
#    print(reader2)
    
    training_df = pd.concat([reader1, reader2])
    shuffled_training_df = training_df.sample(frac=1).reset_index(drop=True)  
    
    with open('./Data/train_test_2/both_wifi(train).csv', 'w') as file:
        shuffled_training_df.to_csv(file, sep=',')
    

    # combine 2 test set
    csvFile1 = open("./Data/train_test_2/simon_wifi(test).csv", "r")
    csvFile2 = open("./Data/train_test_2/xing_wifi(test).csv", "r")
    
    reader1 = pd.read_csv(csvFile1, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    reader2 = pd.read_csv(csvFile2, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
#    print(reader1)
#    print(reader2)
    
    test_df = pd.concat([reader1, reader2])
    shuffled_test_df = test_df.sample(frac=1).reset_index(drop=True)  
    
    with open('./Data/train_test_2/both_wifi(test).csv', 'w') as file:
        shuffled_test_df.to_csv(file, sep=',')
# -----------------------------------------------------------------------------------------





# -111-------------------------------------------------------------------------------------
def combine_three_dataset():
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
    
    # combine 3 training set
    csvFile1 = open("./Data/train_test_"+ ss +"/simon_wifi(train).csv", "r")
    csvFile2 = open("./Data/train_test_"+ ss +"/xing_wifi(train).csv", "r")
    csvFile3 = open("./Data/train_test_"+ ss +"/example_wifi(train).csv", "r")
    
    reader1 = pd.read_csv(csvFile1, usecols=['t', 'x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
#    print(reader1)
    del reader1['t']
    reader2 = pd.read_csv(csvFile2, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
#    print(reader2)
    reader3 = pd.read_csv(csvFile3, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    
    training_df = pd.concat([reader1, reader2,reader3])
    shuffled_training_df = training_df.sample(frac=1).reset_index(drop=True)  
    
    with open("./Data/train_test_"+ ss +"/both3_wifi(train).csv", 'w') as file:
        shuffled_training_df.to_csv(file, sep=',')
    del csvFile1, csvFile2, csvFile3, reader1, reader2, reader3
    
    
    # combine 3 test set
    csvFile1 = open("./Data/train_test_"+ ss +"/simon_wifi(test).csv", "r")
    csvFile2 = open("./Data/train_test_"+ ss +"/xing_wifi(test).csv", "r")
    csvFile3 = open("./Data/train_test_"+ ss +"/example_wifi(test).csv", "r")
    
    reader1 = pd.read_csv(csvFile1, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    reader2 = pd.read_csv(csvFile2, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    reader3 = pd.read_csv(csvFile3, usecols=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    
    test_df = pd.concat([reader1, reader2, reader3])
    shuffled_test_df = test_df.sample(frac=1).reset_index(drop=True)  
    
    with open("./Data/train_test_"+ ss +"/both3_wifi(test).csv", 'w') as file:
        shuffled_test_df.to_csv(file, sep=',')
# -----------------------------------------------------------------------------------------







# 顺序执行以下3个函数，分别处理simon的数据，xingji的数据，然后再合并两个数据集
        
# processing simon's data
#traverse_all("./Archive")
        
# processing xing's data
#extract_saved_xing()

# combine simon's data and xing's data
#combine_two_dataset()

# combine simon's data, xing's data and 
combine_three_dataset()
        
print("Finished traverse all.")