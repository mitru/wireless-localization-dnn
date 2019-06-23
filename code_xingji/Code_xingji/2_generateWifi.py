import os
import xml.dom.minidom
import collections
import numpy as np
import re

import pandas as pd
from scipy import interpolate
from collections import Counter

# =============================================================================
# this file generate interpolated(using0) and down sampled(1s generate 1 sample, interval is 100ms) wifi instances
# 
# =============================================================================


#features_dim = 102
features_dim = 135
#features_dim = 164

# -111---------------------------------------------------------------------------------------
# Read the 102 distinct APs from Xingji's data to 'WIFI_DICT'
# "./Data/wifi_id1.txt": distinct APs extracted from Xingji's data   
# "./Data/wifi_id2.txt": distinct APs extracted from Simon's data   **applied**

def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict
# ----------------------------------------------------------------------------------------

wifi_filename = "./Data/wifi_id2.txt"

WIFI_DICT = read_ap_to_dict(wifi_filename)




# =============================================================================
# def read_ap_to_dict(filename):
#     ap_dict = {}
#     with open(filename) as file:
#         for line in file:
#             elements = re.split(r'[\s]', line.strip())
#             ap_dict[elements[0]] = int(elements[1])
#     return ap_dict
#  
# wifi_filename1 = "./Data/wifi_id1.txt"
# wifi_filename2 = "./Data/wifi_id2.txt"
# w1 = read_ap_to_dict(wifi_filename1)
# w2 = read_ap_to_dict(wifi_filename2)
#  
# z = dict(Counter(w1)+Counter(w2))
# sorted_z = sorted(zip(z.values(), z.keys()))[::-1]
# 
# WIFI_DICT = collections.OrderedDict()
# i = 0
# for v,k in sorted_z:
#     i += 1
#     WIFI_DICT[k] = (str(v), str(i))
# 
# =============================================================================


# ----4444------------------------------------------------------------------------------------
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
# ----4444------------------------------------------------------------------------------------








# ---333------------------------------------------------------------------------------------
# Generate the interpolated location according to the input time list<t_list>
# returns only 2 columns x-lng, y-lat
def get_interpolated_loacation(prefix, t_list):
    csvFile = open("./Archive/"+prefix+".csv", "r")
    reader = pd.read_csv(csvFile)
    print(reader)
    del reader['lat']
    del reader['Lng']
    del reader['x']
    del reader['y']

#    time_serial = reader["Time"]
    
    f_lat = interpolate.interp1d(reader['Time'], reader['lat1'], kind='linear')
    f_lng = interpolate.interp1d(reader['Time'], reader['Lng1'], kind='linear')
    c23 = []
    
    for tt in t_list:
        interpolated_lat = f_lat(tt).ravel()[0]
        interpolated_lng = f_lng(tt).ravel()[0]
        lat_lng = [interpolated_lat, interpolated_lng]
        c23.append(lat_lng)

    c23 = np.array(c23)
    c23[:, 0:] = latlng_to_cor(c23[:, 0:])
    csvFile.close()
    return c23
    

# Generate the interpolated location according to the input time list<t_list>
# returns 3 columns t, x-lng, y-lat
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

# ---333--------------------------------------------------------------------------------------
# The units of 't_list' is milisecond, starting from the t1, which means it corresponds to 0
def filling(t_list, c, nrows, t1, t2, is_tyx):
    
    f = np.zeros((nrows, c.shape[1]))
#    start_time = c[0,0]
#    end_time = c[-1,0]
    
    time_list = np.array(np.arange(t1, t2+0.001, 0.001))
    
    if is_tyx:
        f[:,0] = time_list
    
    for tt, xx in zip(t_list, c):
        if t1 + tt/1000 <= t2:
            f[tt, :] = xx
    
    return f
# ----------------------------------------------------------------------------------------









# 生成0.001s一个插值的wifi样本
# --222--------------------------------------------------------------------------------------
def scan_wifi_and_write(file_name):
    # 记录一个文件中的所有记录的wifi
    wifi_list = list()
    t_list = list()

    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement

    wr_list = root.getElementsByTagName('wr')
    a_list = root.getElementsByTagName('a')
    m_list = root.getElementsByTagName('m')

    start = min(a_list[0].getAttribute("t"), m_list[0].getAttribute("t"))

    for item, i in zip(wr_list, range(len(wr_list))):  # for each time step

        t = item.getAttribute("t")
        t = int(t) - int(start)
        # wifi_record用于记录此时间步的wr向量
        wifi_record = []  # wifi_record is a list, which contains len(wifi_dict) elements
        for _ in range(len(WIFI_DICT.keys())):
            wifi_record.append(0)

        for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
            if j % 2:
                ap_id = item.childNodes[j].getAttribute("b")
                ap_v = item.childNodes[j].getAttribute("s")
                if ap_id in WIFI_DICT.keys():
                    index = int(WIFI_DICT[ap_id][1]) - 1
                    wifi_record[index] = int(ap_v)

        # wifi_list表示一个xml文件中所有的wifi
        wifi_list.append(wifi_record)
        t_list.append(t)

    # 处理完整个xml文件了，所需信息都记在wifi_list里了
    prefix = file_name.split("/")[-1].split(".")[0]

    data_shape = (len(wifi_list), len(WIFI_DICT.keys()))
    wifi_records = np.reshape(wifi_list, data_shape)
    
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

    # 时间t和位置标签xy,需要注意t_llist时间范围长于c_123,这是因为按照simon收集数据的方式，秒表停了之后依然会有wifi scan, 所以c_123只算了位置1和位置4之间，有位置可循的额wifi记录
    c_123 = get_label_list(prefix, t_list)
    n_rows = int(1000*t2-1000*t1)+1
    # 填充全为0的大的f_123
    f_123 = filling(t_list, c_123, n_rows, t1, t2, True)

    # wifi数组
    wifi_array = normalize_wifi_inputs(wifi_records)
    c_4 = np.zeros((c_123.shape[0], len(wifi_record)))      # 注意这里用的是wifi_record的长度，而不是wifi_records
    c_4 = np.array(wifi_array)
    # 填充全为0的大的f_4
    f_4 = filling(t_list, c_4, n_rows, t1, t2, False)

    # Plot wifi collecting frequency
    # # 将wifi_array的内容根据t_list填充进df4中
    # cc = np.zeros((c_4.shape[0], 1))
    # for t, wifi in zip(t_list, wifi_array):
    #     if t >= 1000*c_123[0][0] and t <= 1000*c_123[-1][0]:
    #         ind = int(t - 1000*c_123[0, 0])
    #         cc[ind] = 1
    # plt.plot(cc)
    # plt.show()
    # plt.savefig(prefix+".png")
    

# =============================================================================
#     # ----------填充前后共1s范围内的区间----- *** 想一下！*** ---------------------
#     head_list = list()
#     tail_list = list()
#     # 加入第一个区间的起点到head_list中
#     head_list.append(t_list[0]-500)
# 
#     for i, t_m in enumerate(t_list[:-1]):
#         t_t = t_m+500   # t_tail
#         tn_s = t_list[i+1]-500  # tn_s:下一个t_m对应区间的开头
# 
#         if t_t < tn_s:  # 当前区间和下一个区间不相交
#             tail_list.append(t_t)
#             head_list.append(tn_s)
#         else:   # 当前区间和下一个区间相交
#             mid = (tn_s + t_t)/2
#             tail_list.append(mid)
#             head_list.append(mid)
# 
#     # 加入最后一个区间的终点到tail_list中
#     tail_list.append(t_list[-1]+500)
# 
#     # 按照head_list和tail_list填充c_4
#     for head, tail, wifi in zip(head_list, tail_list, wifi_array):
#         edge1 = 1000 * c_123[0][0]
#         edge2 = 1000 * c_123[-1][0]
#         if head >= edge1 and tail <= edge2:
#             head = int(head - edge1)
#             tail = int(tail - edge1)
#             c_4[head:tail, :] = np.tile(wifi, tail-head).reshape(tail-head, len(wifi))
#     # ----------------------------------------------------------------
# =============================================================================

    total_content = np.concatenate((f_123, f_4), axis=1)

    # write <t,x,y,wifi 102>  to '.csv' file
    total_df = pd.DataFrame(total_content, columns=['t', 'lat', 'lng']+['ap'+str(i) for i in range(len(wifi_record))])
    with open('./wifi_2/wifi_'+prefix+'.csv', 'w') as writeFile:
        total_df.to_csv(writeFile, index=False, sep=',')
        
    print("current sample size: {}\n".format(np.shape(total_content)))
# ----------------------------------------------------------------------------------------









# ---333-------------------------------------------------------------------------------------
#得到"./Archive/"+suffix+".xml"中所有wifi scan的时间
def get_time_list(suffix):
    
    file_name = "./Archive/"+str(suffix)+".xml"
    
    # 记录一个文件中的所有记录的wifi的时间
    t_list = list()

    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement

    wr_list = root.getElementsByTagName('wr')
    a_list = root.getElementsByTagName('a')
    m_list = root.getElementsByTagName('m')

    start = min(a_list[0].getAttribute("t"), m_list[0].getAttribute("t"))

    for item, i in zip(wr_list, range(len(wr_list))):  # for each time step

        t = item.getAttribute("t")
        t = int(t) - int(start)
        t_list.append(t)
    
    return t_list
# ----------------------------------------------------------------------------------------


# --222--------------------------------------------------------------------------------------
def down_to_middle(path):
    
    file_name = path.split("/")[-1]
    suffix = re.search(r'\d+', file_name).group(0)
    
#    get time list where wifi scan are available
    t_list = get_time_list(suffix)
    csvFile = open(path, "r")
    origin_df = pd.read_csv(csvFile)
    
#    根据origina_df中的第一个t和最后一个t来计算本文件down sample后所应得的样本个数，用来与inertial sensor的模型对齐
    sample_num =  int(((np.array(origin_df.iloc[-1:, 0:1])-np.array(origin_df.iloc[0:1, 0:1]))[0][0])*10 - 10 + 1)
    sampled_array = np.zeros((sample_num, features_dim+3))
    start_t = np.array(origin_df.iloc[0:1, 0])[0]
    
#    以下代码块实现：选中一个1s的区间，以mid为中心，前后各半秒，为该区间找到离mid最近的一条非零记录
#   进行sample_num次循环，每次填充一个sample,即sampled_array中的一行
    for i in range(sample_num):
        mid = 500+i*100
#       筛选出mid前后各0.5秒的区间内有非零记录的时间，存入tt_list
        tt_list = list()
        for t in t_list:
            if abs(t-mid) <= 500:
                tt_list.append(t)      
#       前后各0.5秒的时间内有非零值
        if len(tt_list) > 0:
            # 在区间[ind1，ind2]中，找离mid最近的非零值
            for tt in tt_list:
                cur_dis = mid - tt
                
                if cur_dis > 0: # 仍然没到time_end,扫描略过之前的所有wr,记住time_end的前一个wr
                    pre_dis = cur_dis
                    t_selected = tt
                    continue
                else:                   # 读到了time_end后的那个wr，比较前面一个wr和后面一个wr，哪个更近
                    try:    # 如果mid之前有非零值
                        # before "time_end" is the closest
                        if pre_dis + cur_dis < 0:
                            tt = mid - pre_dis
                        # after "time_end" is the closest
                        else:
                            tt = mid - cur_dis
                        t_selected = tt
                        break
                    except UnboundLocalError:   #如果mid之前没有非零值
                        t_selected = tt
#           这是sample_array的第i个sample,给其赋值
            sampled_array[i, :] = np.array(origin_df.iloc[int(t_selected):(int(t_selected)+1), :])
        sampled_array[i, 0:1] = start_t + mid/1000
        
    sampled_array[:, 1:3] = get_interpolated_loacation(suffix, sampled_array[:, 0]) 
    
    
    sampled_df = pd.DataFrame(sampled_array, columns=['t', 'x-lng', 'y-lat']+['ap'+str(i) for i in range(features_dim)])

    # 生成每0.1s一个的wifi样本
    with open("./wifi_2/sampled/sampled"+suffix+'.csv', 'w') as writeFile:
        sampled_df.to_csv(writeFile, index=False, sep=',')

    print("current sample size: {}\n".format(np.shape(sampled_array)))
    
#    return sampled_df
# ----------------------------------------------------------------------------------------





# 产生0.001s一个wifi样本
# -111---------------------------------------------------------------------------------------
def traverse_all(path):
    '''
    traverse all xml files and get all wifi scans
    convert them into standard input and write those 't' & 'wifi' data into file
    :return:
    '''
    dirs = os.listdir(path)
    for dd in dirs:
        if dd.endswith('xml'):
            f_id = os.path.join(path, dd)
            print("processing... ", f_id)
            scan_wifi_and_write(f_id)
# ----------------------------------------------------------------------------------------

# 从0.001s一个wifi样本变为0.1s一个wifi样本
# -111---------------------------------------------------------------------------------------
def traverse_interpolate_wifi(path):
    '''
    traverse all csv files and generate the standard output & input to alighed with Simon's model
    convert them into standard input and write those 't' & 'y,x' & 'wifi' data into file
    :return:
    '''
    dirs = os.listdir(path)
    for dd in dirs:
        if dd.endswith('csv'):
            f_id = os.path.join(path, dd)
            print("processing... ", f_id)
            down_to_middle(f_id)
# ----------------------------------------------------------------------------------------



#以下两个函数可以分开顺序执行
#traverse_all("./Archive")

traverse_interpolate_wifi("./wifi_2/")








