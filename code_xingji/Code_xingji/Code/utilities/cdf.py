import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

import statsmodels.api as sm
north_west = (55.945139, -3.18781)  # A
south_east = (55.944600, -3.186537)  # B
X_GRID_NUM = 40
Y_GRID_NUM = 30


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


def transfer_error_in_meters(data, flag):
    # first 2 column of data is target, last 2 column of data is prediction    
    # flag is True, when neural net outputs 1 value(classification)
    if flag:
        index_xy = np.zeros((np.shape(data)[0], np.shape(data)[1] * 2))
        lnglat = np.zeros((np.shape(data)[0], np.shape(data)[1] * 2))

        # target
        index_xy[:, 0] = data[:, 0] // X_GRID_NUM
        index_xy[:, 1] = data[:, 0] % X_GRID_NUM
        # prediction
        index_xy[:, 2] = data[:, 1] // X_GRID_NUM
        index_xy[:, 3] = data[:, 1] % X_GRID_NUM

        delta_lng = (south_east[1] - north_west[1]) / X_GRID_NUM
        delta_lat = (north_west[0] - south_east[0]) / Y_GRID_NUM

        # target lng-x
        lnglat[:, 0] = north_west[1] + index_xy[:, 0] * delta_lng
        # target lat-y
        lnglat[:, 1] = north_west[0] - index_xy[:, 1] * delta_lat

        # prediction lng-x
        lnglat[:, 2] = north_west[1] + index_xy[:, 2] * delta_lng
        # prediction lat-y
        lnglat[:, 3] = north_west[0] - index_xy[:, 3] * delta_lat

        errors = np.zeros((np.shape(lnglat)[0], 1))

        for item, i in zip(lnglat, range(np.shape(errors)[0])):
            errors[i, 0] = get_distance([item[0], item[1]], [item[2], item[3]])

    # flag is False, when neural net outputs 2 value(regression)
    # the passed in "data" has 4 cols in this case
    else:
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


# \\wrong version, do not use
# def plot_cdf(mark):

# read neural(target & prediction) from "test_output_{}.txt",
# and write error in meters into file "e_{}.txt"
def save_error_in_meters(mark, which_comparison, fn_suffix):
    '''

    :param mark: 1 - classification, 2 - regression
    :param fn_suffix: identify which model has been used(classification/regression, what is the hidden layer's structure)
    :return: no return, but write error in meters in file "./interim_output/e_{}.txt"
    '''
    fn = '../../Experiments/comparison{}/test_output_{}.txt'.format(which_comparison,fn_suffix)
    # experiment = ["classification", "regression"]
    target_and_neural_out = np.loadtxt(fn, delimiter=',')

    # classification
    if mark == 1:
        error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=True)
        with open("../../Experiments/comparison{}/e_{}.txt".format(which_comparison, fn_suffix), "w") as f:
            np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    # regression
    else:
        error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=False)
        with open("../../Experiments/comparison{}/e_{}.txt".format(which_comparison, fn_suffix), "w") as f:
            np.savetxt(f, error_in_meters, delimiter=",", newline='\n')





# *** important ！！！***
def temp_error_in_meter_plotcdf():
#    fn = '../../interim_output/test_output_tree.txt'
#    fn = '../../interim_output/test_output_knn.txt'
    fn = '../../Graph/102_svm/test_output_svm2.txt'
    # experiment = ["classification", "regression"]
    target_and_neural_out = np.loadtxt(fn, delimiter=',')

    error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=True)
#    with open("../../interim_output/e_svm.txt", "w") as f:
    with open("../../Graph/102_svm/e_svm2.txt", "w") as f:
        np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    fig = plt.figure()
#    data = np.loadtxt("../../interim_output/e_svm.txt")
    data = np.loadtxt("../../Graph/102_svm/e_svm2.txt")
    num_examples =  np.shape(data)[0]
    cdf_plot(data, "SVM Regressor", 100, num_examples)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("../../Graph/102_svm/CDF_knn2.png")
    print("\n")


# correct implementation version
def cdf_plot(data, name, number, line_num):
    ecdf = sm.distributions.ECDF(data)

    # x = np.linspace(min(data), max(data), number)
    lower_bound = ecdf.x[int(line_num*0.1)]
    upper_bound = ecdf.x[int(line_num*0.9)]
    x = np.linspace(lower_bound, upper_bound, number)
    y = ecdf(x)

    print("x ~ [{}, {}]".format(round(ecdf.x[0], 3), round(ecdf.x[line_num], 3)))
    print("mid_x={}\t\t{}\n".format(round(ecdf.x[line_num//2], 3), name))

    # plt.step(x, y, label=name)
    plt.plot(x, y, label=name)

# 比较用不同wifi模型得出来的数据训练得到的不同top模型的表现
# plot 3 cdf of 3 top models, that is 102-d, 135-d, 164-d wifi models output with the sensors model output
def main1():
#    fn_suffix_list = ["102_top(simon_data)", "135_top", "164_top"]
    fn_suffix_list = ["102_top3", "135_top3", "164_top3"]
#    fn_suffix_list = ["102_top3", "135_top3", "164_top3", "102_top(simon_data)", "135_top", "164_top"]

    fig = plt.figure()
    plt.title("Comparison of top models (errors of 2 top fc-layers)")

#    legend_list = ["L1_input from 102-d wifi model", "L2_input from 135-d wifi model", "L3_input from 164-d wifi model"]
    legend_list = ["L1_input from 102-d wifi model(new data added)", "L2_input from 135-d wifi model(new data added)", "L3_input from 164-d wifi model(new data added)"]
#    legend_list = ["L1_input from 102-d wifi model(new data added)", "L2_input from 135-d wifi model(new data added)", "L3_input from 164-d wifi model(new data added)", "L4_input from 102-d wifi model", "L5_input from 135-d wifi model", "L6_input from 164-d wifi model"]
    for suffix, model_name in zip(fn_suffix_list, legend_list):
        data = np.loadtxt("../../Graph/{}/e(top).txt".format(suffix))
        line_num = np.shape(data)[0]
        cdf_plot(data, model_name, 100,line_num)

    plt.legend(loc=8, bbox_to_anchor=(0.57, 0.02), borderaxespad=0.)
    plt.show()
    fig.savefig("../../CDF1_1.png")
    print("\n")


# 比较单独的wifi模型
def main2():
#    fn_suffix_list = ["102_original_1", "102_restart", "135_restart", "164_restart"]
    fn_suffix_list = ["102_restart3", "135_restart3", "164_restart3"]
#    fn_suffix_list = ["102_restart3", "135_restart3", "164_restart3", "102_original_1", "102_restart", "135_restart", "164_restart"]

    fig = plt.figure()
    plt.title("Comparison of original model and restart models (errors of wifi models)")

#    legend_list = ["L1_102-d original wifi model", "L2_102-d restart wifi model", "L3_135-d restart wifi model", "L4_164-d restart wifi model"]
    legend_list = ["L1_102-d restart wifi model(new data added)", "L2_135-d restart wifi model(new data added)", "L3_164-d restart wifi model(new data added)"]
#    legend_list = ["L1_102-d restart wifi model(new data added)", "L2_135-d restart wifi model(new data added)", "L3_164-d restart wifi model(new data added)", "L4_102-d original wifi model", "L5_102-d restart wifi model", "L6_135-d restart wifi model", "L7_164-d restart wifi model"]
    sum = 0
    for suffix, model_name in zip(fn_suffix_list, legend_list):
        sum = sum + 1
#        if sum == 1:
        if sum != 4:
            data = np.loadtxt("../../Graph/{}/e(restart).txt".format(suffix))
        else:
            data = np.loadtxt("../../Graph/{}/e(original).txt".format(suffix))
        line_num = np.shape(data)[0]
        cdf_plot(data, model_name, 100,line_num)

    plt.legend(loc=8, bbox_to_anchor=(0.60, 0.02), borderaxespad=0.)
    plt.show()
    fig.savefig("../../CDF2_2.png")
    print("\n")
    

# 比较各阶段模型的表现
def main3():
    fn_suffix_list = ["wifi3", "sensor", "top3"]
#    fn_suffix_list = ["wifi", "top"]
#    fn_suffix_list = ["wifi3", "top3"]
#    fn_suffix_list = ["wifi3", "top3", "wifi", "top"]
    
    # save sensor errors
#    neural_out = np.loadtxt(open('../../wifi_3/simon_output/pre14.csv','rb'), delimiter=',', skiprows=1)
#    target = np.loadtxt(open('../../wifi_3/xing_output/output14.csv','rb'), delimiter=',')
#    neural_out = neural_out[:, 1:]
#    target = target[:,0:2]
#    target_and_neural_out = np.hstack((target, neural_out))
#
#    error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=True)
#    with open("../../Graph/comparison/e(sensors).txt", "w") as f:
#        np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    # start plotting the cdf
    fig = plt.figure()
    plt.title("Comparison of different stages errors")

#    legend_list = ["L1_wifi model's errors", "L2_sensors model's errors", "L3_top model's errors"]
    legend_list = ["L1_wifi model's errors(new data added)",  "L2_sensor model's errors", "L3_top model's errors"]
#    legend_list = ["L1_wifi model's errors(new data added)",  "L2_top model's errors(new data added)", "L3_wifi model's errors(original)",  "L4_top model's errors"]
    for suffix, model_name in zip(fn_suffix_list, legend_list):
        data = np.loadtxt("../../Graph/comparison/e({}).txt".format(suffix))
        line_num = np.shape(data)[0]
        print(line_num)
        cdf_plot(data, model_name, 100,line_num)

    plt.legend(loc=8, bbox_to_anchor=(0.62, 0.02), borderaxespad=0.)
    plt.show()
    fig.savefig("../../CDF3.png")
    print("\n")
    

# 比较knn，决策树和3层神经网络的表现
def main4():
#    fn_suffix_list = ["knn", "wifi3"]
    fn_suffix_list = ["knn", "tree", "wifi3"]
    fig = plt.figure()
    plt.title("Comparison of different models")

    legend_list = ["L1_K Neighbors Regressor", "L2_Decision Tree Regressor", "L3_Deep Neural Network"]
    for suffix, model_name in zip(fn_suffix_list, legend_list):
        data = np.loadtxt("../../Graph/comparison/e({}).txt".format(suffix))
        line_num = np.shape(data)[0]
        cdf_plot(data, model_name, 100,line_num)

    plt.legend(loc=8, bbox_to_anchor=(0.62, 0.02), borderaxespad=0.)
    plt.show()
    fig.savefig("../../CDF4.png")
    print("\n")



def main0():
    
    fn_suffix_list = ["knn1", "knn2", "knn0"]
    fig = plt.figure()
    plt.title("Comparison of knn trained/tested on different data set")

    legend_list = ["L1_knn(new collected data)", "L2_knn(simon)", "L3_knn(xingji)"]
    for suffix, model_name in zip(fn_suffix_list, legend_list):
        data = np.loadtxt("../../Graph/comparison/e({}).txt".format(suffix))
        line_num = np.shape(data)[0]
        cdf_plot(data, model_name, 100,line_num)

    plt.legend(loc=8, bbox_to_anchor=(0.62, 0.02), borderaxespad=0.)
    plt.show()
    fig.savefig("../../cdf_knn.png")
    print("\n")

    

#def temp_func():
#    error_in_meters = pd.read_csv("../../pre14.csv")
#    
##    with open("../../interim_output/e_svm.txt", "w") as f:
#    with open("../../Graph/comparison/e_sensor.txt", "w") as f:
#        np.savetxt(f, error_in_meters, delimiter=",", newline='\n')
    
    
if __name__ == '__main__':
    
#    main0()
#    main1()
#    main2()
    main3()
#    main4() 
    

#     temp_error_in_meter_plotcdf()
