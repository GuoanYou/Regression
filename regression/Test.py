#coding=utf-8
import os
import errno
import random
import math
#输入测试集数据
def input_testdata(file_name=None):
    """input the random data from external file """
    x_list = []
    y_list = []
    float_list = []
     
    if file_name is None:
        file_object = open('','r')
    else:
        file_object = open(file_name,'r')
        
        
    for line in file_object:
        #line = line.rstrip(',')
        str_list = line.split(',')    
        y_list.append(float(str_list.pop()))
        
        for eachstr in str_list:
            float_list.append(float(eachstr))
         
        x_list.append(float_list)
        float_list = []   
     
    #print "input_testdata:"
    #print "x_list:", x_list
    #print "y_list", y_list 
    file_object.close() 
    return x_list, y_list   
#输入Theta数据
def input_thetadata(file_name=None):
    """input the random data from external file """
    theta_list = []
    float_list = []
     
    if file_name is None:
        file_object = open('','r')
    else:
        file_object = open(file_name,'r')
               
    for line in file_object:
        #line = line.rstrip(',')
        str_list = line.split(':')    
        theta_list.append(float(str_list.pop())) 
    #print "input_thetadata:"
    #print "x_list:", theta_list
    #print "y_list", y_list 
    file_object.close() 
    return theta_list
	
thetaNum=71
testx,testy=input_testdata("testdata.txt")
theta_list=input_thetadata("thetasto.txt")
#求RMSE
def rmse(testx,testy,theta_list,thetaNum):
	hy=0.0
	tmp=0.0
	inputx_num = len(testx)
	for j in range(inputx_num):
		for l in range(thetaNum):
			hy+=testx[j][l]*theta_list[l]
		tmp+=pow(hy-testy[j],2)
		print hy,' '
		hy=0.0
	rmse=math.sqrt(tmp/inputx_num)
	return rmse
#打印RMSE
rmse=rmse(testx,testy,theta_list,thetaNum)
print "RMSE="
print rmse
