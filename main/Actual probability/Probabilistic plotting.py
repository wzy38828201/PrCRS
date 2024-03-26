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

n_ = ['name','age','Relative date','CRS','Chance 1','Chance 2']#

a = pd.read_excel(r'main/data/Actual probability/Probabilistic_correspondence.xlsx')
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
#    #"font.family":'Times New Roman', 
#    "font.size":30
##     "mathtext.fontset":'stix',
#}
#rcParams.update(config)

for yyy in hhj:
    a_1 = a[a['name'] == yyy]
    
    #图1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(a_1['Relative date'], a_1['Chance 1'],color='red',label="Chance 1")
    ax1.plot(a_1['Relative date'], a_1['Chance 2'],color='blue',label="Chance 2")
    #CRS在严重以上的有哪些
    li = []
    for ii,jj in enumerate(a_1['CRS']):
        if int(jj)>0: 
            for mm,nn in enumerate(a_1['Relative date']):
                if mm==ii:
                    li.append(int(nn))

    
    lis = []
    for a111 in a_1['CRS']:
        if str(a111)!='nan':
            lis.append(a111)
    
    for tt in li:
        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
        plt.plot([tt,tt],[(min(lis)-0.0001),(max(lis)+0.0001)],linewidth = 1,color='black')
    
#    for tt in li1:
#        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
#        plt.plot([tt,tt],[0.2,0.9],linewidth = 2,color='red')

    for tt in a_1['CRS']:
        #plt.plot([tt,tt],[(min(np.log10(lis))-0.1),(max(np.log10(lis))+0.1)],linewidth = 1,color='black')
        plt.plot([0,0],[0,1],linewidth = 1,color='white')

    for aa1,bb1 in zip(a_1['Relative date'],a_1['Chance 1']):
        plt.text(aa1, bb1+0.0001, '%.3f' % bb1, ha='center', va= 'bottom',fontsize=12)
    for aa2,bb2 in zip(a_1['Relative date'],a_1['Chance 2']):
        plt.text(aa2, bb2+0.0001, '%.3f' % bb2, ha='center', va= 'bottom',fontsize=12)

    ax1.legend(loc='upper left')
    plt.title(str(yyy),fontdict={'size': 25})
    plt.xticks(fontproperties = 'Times New Roman', size = 30)
    plt.yticks(fontproperties = 'Times New Roman', size = 30)
    plt.xlabel('Days',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)
    plt.ylabel('Probability',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)
#    plt.show()
    plt.savefig('main/data/Actual probability/Probabilistic correspondence/Probability_graph//'+str(yyy)+'.png',dpi = 300,bbox_inches='tight')









    