#coding=utf-8

#随机梯度下降---多元线性回归算法实现
import os
import errno
import random
import math
#生成测试用随机数据。然并卵
def generate_data(para_list, length):
    """generate psedo data for the linear regression test """
    ##check the input parameter
    index = range(len(para_list))
    number = xrange(length)
    list1=[]
    target=0.0
    ft_list=[]
    tgt_list=[]
    
    for j in number:
        for i in index:
            list1.append(random.randint(-1000, 1000))
        ##list1[0] = 1
        for i in index:
            target += para_list[i] * list1[i]
        
        ft_list.append(list1)
        tgt_list.append(target)
        list1 = []
        target = 0.0
    
    ##print "feature list:", ft_list
    ##print "target list:", tgt_list
    
    return ft_list, tgt_list
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

#将求得的Theta值输出到文档中。以待测试用	
def save(filename, contents):
	c_len = len(contents)
	fh = open(filename, 'w')
	count = xrange(c_len)
	
	for i in count:
		fh.write(str(contents[i])+'\n')
		
	fh.close()
##就是hypyhosis。 result += x_list[i] * para_list[i] 然后返回所求结果
def _hypyhosis(x_list,para_list):
    """according to the feature and para list to calcualte the value """
    
    number = len(x_list)
    
    if(number != len(para_list)):
        raise ValueError('illeal para and sample feature numbers ' %(name,))
    result=0.0
    
    index=range(number)
    for i in index:
        result += x_list[i] * para_list[i]

    ##print "result in _hypyhosis", result
    return result
#迭代梯度下降更新theta（para），然后返回更新后的theta（就是减去一长串式子，与批量梯度下降区别在于每次只用一条数据）
def _single_step(x_list, y, para_list, learn_rate):
    """One single step for the gradient decent """
    ##check the input parameter
    result = para_list
    index = range(len(para_list))
    for i in index:
        result[i] = para_list[i] + (y - _hypyhosis(x_list, para_list)) * x_list[i] * learn_rate
        ##print "step expect, para", i, "is: ", result[i]
    return result
##纯粹就是多了一步 参数：特征    y      Theta集合  学习速率  样例个数
def stochastic_gradient(x_list, y_list, para_list, rate, sample_number):
    """The algorithm for the stochastic gradient decent """
    count = xrange(sample_number)
    result=para_list
    for j in count:
        result = _single_step(x_list[j], y_list[j], result, rate)
    
    return result            


print "Stochastic Gradient Linear Regression Algorithm"



#count = xrange(40)
know_paras = [10.0, -12.0, 10.0, 4.0]
testx, testy = input_data("test.txt")
samp_len = len(testx)
#重要参数：学习速率，需要调节
learn_rate=0.00006
#Theta初始值，全部设为0即可。
paras=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    ##parameters,  h(x) = paras[0] * sampx[0] + paras[1] * sampx[1] + paras[2]*sampx[2]

##testx = sampx
##testy = sampy
##samp_len=5

para_result = paras
#print "inputx:", testx
#print "inputy:", testy
print "sample_len:", samp_len
print "para_result:", para_result
print "learn_rate:", learn_rate


para_result = stochastic_gradient(testx, testy, para_result, learn_rate, samp_len)

para_num = len(para_result)
#输出Theta值。并且保存到文档中以待测试。
for i in range(para_num):
    print "para_result: ", i, ": ", para_result[i]
#output_data(para_result,"thetasto.txt")
save("thetasto.txt",para_result)







