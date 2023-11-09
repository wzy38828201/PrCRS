# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:50:33 2023

@author: lenovo
"""
from matplotlib import rcParams
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='SimHei', weight='bold')
plt.rcParams['axes.unicode_minus'] = False

n_ = ['姓名','年龄','相对日期','CRS','概率1','概率2']#先把”对应概率“里面最右边的两个列名换成概率1和概率2

a = pd.read_excel(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\12个测试\概率对应.xlsx')
a = a[n_]

#将五分类变成二分类
er = []
er1 = ['0','1']
er2 = ['2']
er3 = ['3','4']
for e in a['CRS']:
    if str(e) in er1:
        er.append(0)
    elif str(e) in er2:
        er.append(0)
    elif str(e) in er3:
        er.append(1)
a['CRS'] = er

#a = a.dropna(axis=0, how='any', thresh=None, subset=None,inplace=False)
hhj = a['姓名'].unique().tolist()

#config = {
#    #"font.family":'Times New Roman',  # 设置字体类型
#    "font.size":30
##     "mathtext.fontset":'stix',
#}
#rcParams.update(config)

for yyy in hhj:
    a_1 = a[a['姓名'] == yyy]
    
    #图1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(a_1['相对日期'], a_1['概率1'],color='red',label="概率1")
    ax1.plot(a_1['相对日期'], a_1['概率2'],color='blue',label="概率2")
    #CRS在严重以上的有哪些
    li = []
    for ii,jj in enumerate(a_1['CRS']):
        if int(jj)>0: 
            for mm,nn in enumerate(a_1['相对日期']):
                if mm==ii:
                    li.append(int(nn))

#    #概率在0.3以上的有哪些
#    li1 = []
#    for ii1,jj1 in enumerate(a_1['概率']):
#        if float(jj1)>0.3:
#            for mm1,nn1 in enumerate(a_1['相对日期']):
#                if mm1==ii1:
#                    li1.append(int(nn1))
    
    lis = []
    for a111 in a_1['CRS']:
        if str(a111)!='nan':
            lis.append(a111)
    
    #画严重CRS的竖线
    for tt in li:
        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
        plt.plot([tt,tt],[(min(lis)-0.0001),(max(lis)+0.0001)],linewidth = 1,color='black')
    
#    #画已经预测出来的红线
#    for tt in li1:
#        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
#        plt.plot([tt,tt],[0.2,0.9],linewidth = 2,color='red')

    #让图片的y坐标为[0,1]区间的
    for tt in a_1['CRS']:
        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
        plt.plot([0,0],[0,1],linewidth = 1,color='white')

    for aa1,bb1 in zip(a_1['相对日期'],a_1['概率1']):
        plt.text(aa1, bb1+0.0001, '%.3f' % bb1, ha='center', va= 'bottom',fontsize=12)
    for aa2,bb2 in zip(a_1['相对日期'],a_1['概率2']):
        plt.text(aa2, bb2+0.0001, '%.3f' % bb2, ha='center', va= 'bottom',fontsize=12)

    ax1.legend(loc='upper left')
    plt.title(str(yyy),fontdict={'size': 25})
    plt.xticks(fontproperties = 'Times New Roman', size = 30)
    plt.yticks(fontproperties = 'Times New Roman', size = 30)
    plt.xlabel('Days',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)
    plt.ylabel('Probability',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)
#    plt.show()
    plt.savefig(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\12个测试\概率图\\'+str(yyy)+'.png',dpi = 300,bbox_inches='tight')









    