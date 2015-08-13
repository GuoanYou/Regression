#coding=utf-8

#批量梯度下降---多元线性回归算法实现
import os
import errno
import random
import math
#def generate_data(para_list, length, file_name=None):
#生成测试用随机数据。然并卵
def generate_data(para_list, length):
    """generate psedo data for the linear regression test """
    ##check the input parameter
    index = range(len(para_list))
    number = xrange(length)
    list1 = []
    target = 0.0
    rand_value = 0.0
    #outpt_str = ""
    ft_list = []
    tgt_list = []
    
    #if file_name is None:
    #    file_object = open('randomdata.txt','w')
    #else:
    #    file_object = open(file_name,'w')
        
    for j in number:
        for i in index:
            rand_value = random.randint(-100, 100)
            list1.append(rand_value)
            #outpt_str += "%8.3f " %(rand_value,)
        
        for i in index:
            target += para_list[i] * list1[i]
        
        ft_list.append(list1)
        tgt_list.append(target)
        #outpt_str += "%8.3f\n" %(target,)
        #print "output_str:", outpt_str
        #file_object.write(outpt_str)
        outpt_str = ""
        list1 = []
        target = 0.0
    
    
    #file_object.close()
    return ft_list, tgt_list

#将x_list y_list输出到文件中。然并卵	
def output_data(x_list, y_list, file_name=None):
    """output the random data to external file """
    ft_len = len(x_list)
    tgt_len = len(y_list)
    outpt_str = ""
    if file_name is None:
        file_object = open('randomdata.txt','w')
    else:
        file_object = open(file_name,'w')
        
    if (ft_len != tgt_len):
        raise ValueError('illegal target and feature sample numbers  ' )

    count = xrange(ft_len)
    for i in count:
        for each_item in x_list[i]:
            outpt_str += "%8.3f " %(each_item,)
        
        outpt_str += "%8.3f\n" %(y_list[i],)
        file_object.write(outpt_str)
        outpt_str = ""

    file_object.close()


#输入数据
def input_data(file_name=None):
    """input the random data from external file """
    x_list = []
    y_list = []
    float_list = []
     
    if file_name is None:
        file_object = open('test.txt','r')
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
     
    ##print "input_data:"
    ##print "x_list:", x_list
    ##print "y_list", y_list 
    file_object.close() 
    return x_list, y_list    

##就是hypyhosis。 result += x_list[i] * para_list[i] 然后返回所求结果
def _hypyhosis(x_list,para_list):
    """according to the feature and para list to calcualte the value """
    
    number = len(x_list)
    
    if(number != len(para_list)):
        raise ValueError('illeal para and sample feature numbers ')
    result=0.0
    
    index=range(number)
    for i in index:
        result += x_list[i] * para_list[i]
        
    ##print "result in _hypyhosis", result
    return result
##梯度下降减号右面的一大串式子。
def _para_expect(x_list,y_list, para_list, sample_number, position):
    """calculate the expect adjust value for the specified parameters """
    ##check the input parameter
    result = 0.0
    index = xrange(sample_number)
    for i in index:
        result1 = (y_list[i] - _hypyhosis(x_list[i], para_list)) * x_list[i][position]
        result += result1
        
    ##print "_para_expect:",  result
    return result

#迭代梯度下降更新theta（para），然后返回更新后的theta（就是减去一长串式子）
def _single_step(x_list,y_list, para_list, sample_number, learn_rate):
    """One single step for the gradient decent """
    ##check the input parameter
    result = para_list
    index = range(len(para_list))
    for i in index:
        result[i] = para_list[i] + learn_rate * _para_expect(x_list, y_list, para_list, sample_number, i)
        ##print "step expect, para", i, "is: ", result[i]
    return result

#
##纯粹就是多了一步
#最终函数参数列表： 特征    y     Theta集合  学习速率  样例个数    迭代次数
def batch_gradient(x_list, y_list, para_list, rate, sample_number, count):
    """The algorithm for the stochastic gradient decent """
    count1 = xrange(count)
    result=para_list
    for j in count1:
        result = _single_step(x_list, y_list, result,sample_number, rate)
    
    return result 

#
print "Batch Gradient Linear Regression Algorithm"


#samp_len = 872
count = xrange(40)
know_paras = [12.0, -10.0, 12.0, 3.0]
#testx, testy = generate_data(know_paras, samp_len)
#output_data(testx, testy, "randdata1.txt")
testx=[]
testy=[]
testx, testy = input_data("test.txt")
samp_len=len(testx)
#Theta初始值，全部设为0即可。
paras=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    ##parameters,  h(x) = paras[0] * sampx[0] + paras[1] * sampx[1] + paras[2]*sampx[2]
para_result = paras
#两个重要参数：学习速率和循环次数。收敛的很快。
learn_rate=0.001
loop_count=20


##sampx=[[1.0, 1.1, 2.0],
##       [2.0, 1.2, 2.0],
##       [1.2, -1.0, 2.1],
##       [1.1, 1.2, 1.0],
##       [0.1, 0.5, 0.9]]

##sampy=[3.0, 5.0, 2.1, 1.2, 0.1]
##samp_len=5
##testx=sampx
##testy=sampy

#print "inputx:", testx
#print "inputy:", testy
print "sample_len:", samp_len
print "para_result init:", para_result
print "learn_rate:", learn_rate
print "loop_count:", loop_count

#跑这一下就跑完了。
para_result = batch_gradient(testx, testy, para_result, learn_rate, samp_len, loop_count)

para_num = len(para_result)
hy=0.0
rmse=0.0
inputx_num = len(testx)

#输出Theta具体的值。相当于输出多元线性回归方程了。
for i in range(para_num):
    print "para_result: ", i, ": ", para_result[i]

#下面求对自身的RMSE。并输出。
for j in range(inputx_num):
	for l in range(71):
		hy+=testx[j][l]*para_result[l]
	rmse+=pow(hy-testy[j],2)
	print hy,' '
	hy=0.0
print math.sqrt(rmse/872.0)
	#print 'next'
	#print  testx[j][0]*para_result[0]+testx[j][1]*para_result[1]+testx[j][2]*para_result[2]+testx[j][3]*para_result[3],
    

