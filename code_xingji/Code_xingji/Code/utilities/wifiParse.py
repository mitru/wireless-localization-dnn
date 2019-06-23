#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:53:01 2019

@author: chenxingji
"""

import os
import re
import xml.dom.minidom
import collections
import numpy as np
import h5py
#from collections import Counter
import pandas as pd

# define some constant variables
north_west = (55.945139, -3.18781)
south_east = (55.944600, -3.186537)
num_grid_y = 30  # latitude
num_grid_x = 40  # longitude
max_lat = abs(north_west[0] - south_east[0])  # 0.0006 # 0.0005393
max_lng = abs(north_west[1] - south_east[1])  # 0.002  # 0.001280
delta_lat = max_lat / num_grid_y  # 3e-06
delta_lng = max_lng / num_grid_x  # 1e-05


# global varibles
#feature_DIM = 102  # xingji
#feature_DIM = 135  # simon
#feature_DIM = 164  # both



# 本脚本处理xml数据，写入example_*(*表示wifi维度).h5或example_*(*表示wifi维度).txt文件中, 
# 并生成对应特征维度的 example_wifi(train/test).csv文件在train_test_*(*表示wifi维度)文件夹内

# *****************************************************************************************************
# 2. Read the distinct access point id from file("wifi_filename") into dictionary

# ** VERSION 1 **: 102 dimensions(Xingji)/135 dimensions(Simon)
wifi_filename = "../../Data/wifi_id3.txt"

def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict

WIFI_DICT = read_ap_to_dict(wifi_filename)
feature_DIM = len(WIFI_DICT) 



# =============================================================================
# # ** VERSION 2 **: 164 dimensions
# # read APs from both file, and generate the union set of this two ap file, save into WIFI_DICT
# #def read_ap_to_dict(filename):
# #    ap_dict = {}
# #    with open(filename) as file:
# #        for line in file:
# #            elements = re.split(r'[\s]', line.strip())
# #            ap_dict[elements[0]] = int(elements[1])
# #    return ap_dict
# #
# #wifi_filename1 = "../../Data/wifi_id1.txt"
# #wifi_filename2 = "../../Data/wifi_id2.txt"
# #w1 = read_ap_to_dict(wifi_filename1)
# #w2 = read_ap_to_dict(wifi_filename2)
# #
# #z = dict(Counter(w1)+Counter(w2))
# #sorted_z = sorted(zip(z.values(), z.keys()))[::-1]
# #
# #WIFI_DICT = collections.OrderedDict()
# #i = 0
# #for v,k in sorted_z:
# #    i += 1
# #    WIFI_DICT[k] = (str(v), str(i))
# #    
# #feature_DIM = len(WIFI_DICT) 
# =============================================================================


#def write_ap_to_file(world_wifi, filename):
#    file = open(filename, 'w')
#    for wifi_id, times_rank in zip(world_wifi.keys(), world_wifi.values()):
#        file.write('{}\t{}\t{}\n'.format(wifi_id, times_rank[0], times_rank[1]))
#    file.close()
#
#write_ap_to_file(WIFI_DICT, "../../Data/wifi_id3.txt")


# *****************************************************************************************************
# 3. Pre-processing the background files, generate standard input data
# instantiate a WifiFile object for each background file collected


class WifiFile(object):
    # Class variable
    world_ap_dict = WIFI_DICT
    file_rank = 0

    def __init__(self, file_name):
        # Member variables
        self.wr_dict = collections.OrderedDict()
        self.loc_dict = collections.OrderedDict()
        self.fn = file_name

        # Transfer the data from raw file into internal data structure
        self.first_parse_file(file_name)
        self.sample_num = len(self.loc_dict)
        self.f_inputs = np.zeros((self.sample_num, feature_DIM))
        self.f_outputs = np.zeros((self.sample_num, 5))

        # Filter out(reduce frequency) useful input data according to recorded location
        self.generate_instances()
        self.f_outputs[:, :2] = self.latlng_to_cor(self.f_outputs[:, :2])

        # Save standard input and output into files
#        self.save_overall_txt(feature_DIM)
        self.save_overall_hdf5(feature_DIM)

    def first_parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        loc_list = root.getElementsByTagName('loc')

        print("# wifi record:", wr_list.length)
        print("# loc record:", loc_list.length)

        # 将处理xml得到的loc_list写入内存中的self.loc_dict变量中
        # location(user input)
        for item, i in zip(loc_list, range(len(loc_list))):
            try:
                t = int(item.getAttribute("time"))
                lat = float(item.getAttribute("lat"))
                lng = float(item.getAttribute("lng"))
            except ValueError:
                print('invalid input %d: %s,%s'.format(i, lat, lng))
            self.loc_dict[(i, t)] = (lat, lng)

        # 将处理xml得到的wr_list写入内存中的self.wr_dict变量中
        # wifi record
        # for item, i in zip(wr_list, range(len(wr_list))):  # for each time step 有wr记录的time step
        for item in wr_list:  # for each time step 有wr记录的time step
            t = int(item.getAttribute("t"))
            # print(i, "->", t, len(item.childNodes)//2)
            # ap_list是一个ap的列表，一个ap_list表示一个<wr>，代表一个time step记录下来的一个ap的列表
            ap_list = list()
            for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
                if j % 2:
                    ap = item.childNodes[j].getAttribute("b")
                    s = item.childNodes[j].getAttribute("s")
                    if ap not in self.world_ap_dict.keys():
                        pass
                        # self.world_wifi[ap] = 1
                        # print("{} not in world ap dict".format(ap))
                    else:
                        ap_list.append((ap, s))
            # self.wr_dict[(i, t)] = ap_list
            self.wr_dict[t] = ap_list

    # 将self.wr_dict和self.loc_dict 分别写入self.f_inputs和self.f_outputs
    def generate_instances(self):
        if len(self.loc_dict) == len(self.wr_dict):
            i = 0
            for out, inp in zip(self.loc_dict.values(), self.wr_dict.values()):
                self.f_inputs[i, :] = self.func(inp)
                self.f_outputs[i, :] = np.array(WifiFile.latlng_to_grid(out[0],out[1]))
                i = i + 1
                
    def func(self, ori_input):
        wr = WifiFile.formalize_wr(ori_input)    
        return WifiFile.normalize_wifi_inputs(wr)

    
    # Normalise each APs strength to [0,1]
    @staticmethod
    def normalize_wifi_inputs(wr_inputs):
    
        zero_index = np.where(wr_inputs == 0)
        wr_inputs[zero_index] = -100
    
        max = -40
        min = -100
        wr_inputs = (wr_inputs - min) / (max - min)
    
        return wr_inputs
    
    # convert the original lat&lng to [-1,1] 
    @staticmethod
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
    
    
    @staticmethod
    def formalize_wr(wr):
        ap_num = len(WifiFile.world_ap_dict)  # standard input need same number of input ap
        element = np.zeros(ap_num)
        for ap in wr:
            ap_id = ap[0]
            ap_val = ap[1]
            # find out the index（column index in element） of this ap_id
            ap_index = int(WifiFile.world_ap_dict[ap_id][1]) - 1
            element[ap_index] = ap_val
        return element

    @staticmethod
    def latlng_to_grid(lat, lng):
        y = abs(lat - north_west[0]) // delta_lat  # index start from 0
        x = abs(lng - north_west[1]) // delta_lng  # index start from 0
        index = y * num_grid_x + x  # corresponding index from 0 to 4800
        return float(lat), float(lng), int(x), int(y), int(index)

  
    # 以append的形式写入txt文件    
    def save_overall_txt(self,suffix):
        txt_filename = "../../Data/example_"+str(suffix)+".txt"
        write_text = np.hstack((self.f_outputs, self.f_inputs))
        with open(txt_filename, "ab") as f:     # 以append的形式附加
            np.savetxt(f, write_text, delimiter=",", newline='\n')


    # 以append的形式写入h5文件    
    # write the overall standard input and output into a single "out_in_overall.h5" file
    def save_overall_hdf5(self, suffix):
        h5_filename = "../../Data/example_"+str(suffix)+".h5"
        h5_file = h5py.File(h5_filename, mode='a')
        write_content = np.hstack((self.f_outputs, self.f_inputs))
        h5_file.create_dataset(os.path.basename(self.fn), data=write_content)
        h5_file.close()
        


# Iterate over all the background file in the directory "background"
def iterate(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            fi_d = os.path.join(path, dir)
            if os.path.isdir(fi_d):
                iterate(fi_d)
            else:
                WifiFile(fi_d)
        else:
            pass
            # using "continue" here is the same as using "pass"



def extract_saved_example():
    dataset = h5py.File("../../Data/example_"+str(feature_DIM)+".h5", "r")

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

    # X_Y的3-5列是grid的横纵坐标，和index。也就是X_Y的前5列都对应于位置location
    # get formatted nueral network inputs(standardised wifi signal)- 102 values
    X = X_Y[:, 5:]
    # get formatted and normalised target outputs(x,y)- 2 values
    Y = X_Y[:, :2]
    
    chunk_size = int(0.2 * SAMPLE_NUM)
    Y_test = Y[0:chunk_size, :]
    X_test = X[0:chunk_size, :]

    Y_train = Y[chunk_size:, :]
    X_train = X[chunk_size:, :]
    
    train_data = np.hstack((Y_train, X_train))
    test_data = np.hstack((Y_test, X_test))
    print("train:{}".format(np.shape(train_data)))
    print("test:{}".format(np.shape(test_data)))
    
    train_df = pd.DataFrame(train_data, columns=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    test_df = pd.DataFrame(test_data, columns=['x-lng', 'y-lat']+['ap'+str(i) for i in range(feature_DIM)])
    
    if feature_DIM == 102:
        ss = "1"
    elif feature_DIM == 135:
        ss = "2"
    elif feature_DIM == 164:
        ss = "3"
        
    with open("../../Data/train_test_"+ ss +"/example_wifi(train).csv", 'w') as file:
        train_df.to_csv(file, sep=',')
        
    with open("../../Data/train_test_"+ ss +"/example_wifi(test).csv", 'w') as file:
        test_df.to_csv(file, sep=',')   
    



file1 = "../../Data/example_"+str(feature_DIM)+".h5"
file2 = "../../Data/example_"+str(feature_DIM)+".txt"
if os.path.isfile(file1):
    os.remove(file1)
if os.path.isfile(file2):
    os.remove(file2)

#WifiFile("../../Data/example/1.xml")
iterate("../../Data/example")
extract_saved_example()
