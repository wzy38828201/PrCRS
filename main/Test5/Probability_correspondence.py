# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

#n = ['D-二聚体', '降钙素原', '血浆凝血酶原时间',  '纤维蛋白原', '红细胞', '白细胞', 
#                '中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数', 
#                '单核细胞百分比', '单核细胞计数', '钾', '钙', '葡萄糖', 
#                '甘油三酯', 'γ-谷氨酰转肽酶', 
#                'IL2', 'IL4', 'IL6', 'IL10', 
#                'TNFα', 'IFNγ', 'IL17A', '原幼细胞', '肿瘤负荷','CRS']
n = ['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
    '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
    '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
    '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
     'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A','CRS'] #细胞因子☆√1

a = pd.read_excel(r'\main\data\Records.xlsx')

load_npy = np.load(r"\main\Test5\data5.npy")

load_npy = pd.DataFrame(load_npy)

load_npy.columns = n

#训练集的对应
shu = pd.DataFrame(np.arange(52).reshape(1, 52))
shu.columns = a.columns
for i,j in enumerate(load_npy['中性粒细胞百分比']):
    for m,n in enumerate(load_npy['淋巴细胞百分比']):
        if m==i:
            for m1,n1 in enumerate(load_npy['单核细胞百分比']):
                if m1==i:
                    for m2,n2 in enumerate(load_npy['白细胞']):
                        if m2==i:
                            for m3,n3 in enumerate(load_npy['IL6']):
                                if m3==i:
                                    for m4,n4 in enumerate(load_npy['降钙素原']):
                                        if m4==i:
                                            z1 = a[a['中性粒细胞百分比'] == j]
                                            z2 = z1[z1['淋巴细胞百分比'] == n]
                                            z3 = z2[z2['单核细胞百分比'] == n1]
                                            z4 = z3[z3['白细胞'] == n2]
                                            z5 = z4[z4['IL6'] == n3]
                                            z6 = z5[z5['降钙素原'] == n4]
                                            shu = pd.concat([shu,z6], axis=0)
shu = shu[1:].drop_duplicates()

z = shu
z.index = list(range(len(z)))
b = pd.read_excel(r'\main\Test5\W5总.xlsx')
z[0] = b[0]
z[1] = b[1]
z.to_excel(r'\main\Test5\Probability_correspondence.xlsx',index = 0)
