import os
import numpy as np
import random

import pandas as pd
pd.set_option('display.notebook_repr_html',False)
# 读取xls（绝对路径）
import torch


#xlsx_file2 = r'C:\Users\lenovo\Desktop\zui.xlsx'
xlsx_file2 = r'G:\main\data\CAR-T.xlsx'
data = pd.read_excel(xlsx_file2)
# 将性别变成0和1
data['性别'] = data['性别'].map({'男':0, '女':1})
# 将NaN列 姓名 时间列去掉
data = data.dropna(axis=1)
data = data.drop('姓名', axis=1)

#使每个标签数量均衡
df20 = data
df1_0 = df20[df20['CRS'] == 0].sample(frac=0.6)
df1_1 = df20[df20['CRS'] == 1].sample(frac=0.6)
df1_2 = df20[df20['CRS'] == 2].sample(frac=0.6)
df1_3 = df20[df20['CRS'] == 3].sample(frac=0.6)
df1_4 = df20[df20['CRS'] == 4].sample(frac=0.6)
df2_ = pd.concat([df1_0, df1_1, df1_2, df1_3, df1_4])
df2__ = df20[~df20.index.isin(df2_.index)]

df1__0 = df2__[df2__['CRS'] == 0].sample(frac=0.2)
df1__1 = df2__[df2__['CRS'] == 1].sample(frac=0.2)
df1__2 = df2__[df2__['CRS'] == 2].sample(frac=0.2)
df1__3 = df2__[df2__['CRS'] == 3].sample(frac=0.2)
df1__4 = df2__[df2__['CRS'] == 4].sample(frac=0.2)
df2_z = pd.concat([df1__0, df1__1, df1__2, df1__3, df1__4])#0.2
df2__z = df2__[~df2__.index.isin(df2_z.index)]#0.2
df2 = pd.concat([df2_z, df2__z, df2_])

##新加的
#df2_['CRS'] = df2_['CRS'].map(lambda x: 0 if x < 3 else 1)
#df2__['CRS'] = df2__['CRS'].map(lambda x: 0 if x < 3 else 1)
#
#train, test = df2_, df2__
#
## 以CRS为label，其余为特征，生成数据集
#train_x = train.drop('CRS', axis=1).values
#train_y = train['CRS'].values
#test_x = test.drop('CRS', axis=1).values
#test_y = test['CRS'].values


data2=df2.values
keys2 = list(df2.keys())
print(keys2)
print(data2)

#common_keys = keys2[5:-4]
##至少在5个以上才能识别
# common_keys = ['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
#                 '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
#                 '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
#                 '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
#                  'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A'] #细胞因子☆√1
common_keys = ['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
                '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
                '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
                 'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A'] #细胞因子☆√1

#data1_label2id = {'Severe patients': 0, 'Mild patients':1, 'Follow-up patients': 2, 'Healthy controls': 3}
data2_label2id = {0: 0, 1: 0, 2:0, 3:1, 4:1}

def get_feats(df, common_keys):
    tem = []
    for key in common_keys:
        colfeat = df[key].values
        # colfeat = (colfeat-np.min(colfeat))/(np.max(colfeat)-np.min(colfeat))
        # colfeat = (colfeat-np.min(colfeat))/(np.max(colfeat)-np.min(colfeat))
        tem.append(colfeat)

        # print('df[key].values: ', len(df[key].values), type(df[key].values))
    
    tem = np.array(tem).T

    feats = [x for x in tem]
    # feats = [(x-np.min(x))/(np.max(x)-np.min(x)) for x in feats]
    # feats = [(x-np.mean(x))/(np.std(x) + 1e-4) for x in feats]

    return feats


#test_num = 30
savedir = r'main\data'
##savedir = './data2'
#
data2_feats = get_feats(df2, common_keys)
data2_label = [data2_label2id[x] for x in df2['CRS']]
#print(data2_label)
data2_feats_label = [list(x) + [y] for x, y in zip(data2_feats, data2_label)]
#random.shuffle(data2_feats_label)
train_data2_feats_label = data2_feats_label[int(0.4*len(data2_feats_label)):]#-2*test_num
valid_data2_feats_label = data2_feats_label[int(0.2*len(data2_feats_label)):int(0.4*len(data2_feats_label))]#-2*test_num:-test_num
test_data2_feats_label = data2_feats_label[:int(0.2*len(data2_feats_label))]#-test_num:


print('train_data2_feats_label: ', np.array(train_data2_feats_label).shape)
np.save(os.path.join(savedir, 'data2_train.npy'), np.array(train_data2_feats_label))
np.save(os.path.join(savedir, 'data2_valid.npy'), np.array(valid_data2_feats_label))
np.save(os.path.join(savedir, 'data2_test.npy'), np.array(test_data2_feats_label))

# testset1 = np.load(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\data2\data2_train.npy')
# testset2 = np.load(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\data2\data2_valid.npy')
# testset3 = np.load(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\data2\data2_test.npy')

#train_x = get_feats(train, common_keys)
#test_x = get_feats(test, common_keys)
#data2_label = [data2_label2id[x] for x in df2['CRS']]
#train_x_feats_label = [list(x) + [y] for x, y in zip(train_x, data2_label)]
#test_x_feats_label = [list(x) + [y] for x, y in zip(test_x, data2_label)]
