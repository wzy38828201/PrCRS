# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 09:45:08 2023

@author: lenovo
"""

import shap
import torch
import numpy as np
from sklearn.externals import joblib
#model = joblib.load(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\outputs1\49.pt')

#common_keys = ['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
#                '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
#                '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
#                '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
#                 'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A'] #细胞因子☆√1
#
#
#data1_ckpt = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\outputs1\49.pt'
#model = torch.load(data1_ckpt)['model']
#explainer = shap.TreeExplainer(model) #创建解释器
#
#x_test = np.array(common_keys)
#
#shap_values = explainer.shap_values(x_test) #x_test为特征参数数组 shap_value为解释器计算的shap值
##shap.dependence_plot("参数名称", 计算的SHAP数组, 特征数组, interaction_index=None,show=False)
##data_with_name = pd.DataFrame(x_test) #将numpy的array数组x_test转为dataframe格式。
##data_with_name.columns = ['特征1','特征2','特征3'] #添加特征名称
file = np.load(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\da.npy')