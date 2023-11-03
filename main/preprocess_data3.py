import os
import numpy as np
import random

import pandas as pd
pd.set_option('display.notebook_repr_html',False)
# 读取xls（绝对路径）

# xlsx_file3 = './data3/data3.xlsx'
# df3 = pd.read_excel(xlsx_file3)
# data1=df3.values
# keys3 = list(df3.T.iloc[0].values)
# print(df3)
# # print(df3.T.iloc[0].values)
# # print(keys1)
# # print(data1)
# xlsx_file3_label = './data3/data3_label.xlsx'
# df3_label = pd.read_excel(xlsx_file3_label)
# # print('df3_label: ', df3_label)
#
# patients = list(df3_label['RecordID'].values)
# patients_label = df3_label['COVID_status'].values
# patients2label = {x:y for x, y in zip(patients, patients_label)}
#
# df3_patients = list(df3.keys()[1:])
#
# aval_data = []
# for dp in df3_patients:
#     if dp in patients2label:
#         dp_feat_label = list(df3[dp].values) + [patients2label[dp]]
#         aval_data.append(dp_feat_label)
#
# aval_data = np.array(aval_data)
# print('aval_data: ', aval_data)
# print(aval_data.shape, len(keys3))
# keys3.append('label')
# keys3 = [x.replace('-', '') for x in keys3]
# aval_labels = aval_data[:,-1]
# print('aval_labels:', aval_labels)
#
# df3_aval = pd.DataFrame(data=aval_data, columns=keys3)
# print(df3_aval)

#自己的新数据
df3_aval = pd.read_excel(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\data3\迁移数据用.xlsx')
keys3 = df3_aval.columns.tolist()
keys3.remove('性别')
aval_labels = list(df3_aval['最大严重性'])

#读取要预测的数据，选取其特征为了之后的对比
xlsx_file2 = './data3/data2.xlsx'
df2 = pd.read_excel(xlsx_file2)
data2=df2.values
keys2 = list(df2.keys())
# print(keys2)
# print(data2)

#对比出哪些特征是和要预测的数据一样的特征
common_keys = [key for key in keys3 if key in keys2]
print('common_keys: ', common_keys)

data3_label2id = {'Positive': 0, 'Negative':1}

# def get_feats(df, common_keys):
    # tem = []
    # for key in common_keys:
    #     tem.append(df[key].values)
    #
    # tem = np.array(tem).T
    # # print('tem: ', tem)
    #
    # def toFloat(lst):
    #     return [float(x) for x in lst]
    # feats = [toFloat(x) for x in tem]
    # #feats = [(x-np.min(x))/(np.max(x)-np.min(x)) for x in feats]
    # feats = [(x-np.mean(x))/(np.std(x) + 1e-4) for x in feats]
    #
    # return feats
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

test_num = 10
savedir = './data3'
os.makedirs(savedir, exist_ok=True)

data3_feats = get_feats(df3_aval, common_keys)
data3_label = aval_labels
#data3_feats_label = [list(x) + [data3_label2id[y]] for x, y in zip(data3_feats, data3_label)]
data3_feats_label = [list(x) + [y] for x, y in zip(data3_feats, list(df3_aval['最大严重性']))]
print(data3_feats_label)
random.shuffle(data3_feats_label)
train_data3_feats_label = data3_feats_label[int(0.4*len(data3_feats_label)):]
valid_data3_feats_label = data3_feats_label[int(0.2*len(data3_feats_label)):int(0.4*len(data3_feats_label))]
test_data3_feats_label = data3_feats_label[:int(0.2*len(data3_feats_label))]

print('train_data1_feats_label: ', np.array(train_data3_feats_label).shape)
np.save(os.path.join(savedir, 'data1_train.npy'), np.array(train_data3_feats_label))
np.save(os.path.join(savedir, 'data1_valid.npy'), np.array(valid_data3_feats_label))
np.save(os.path.join(savedir, 'data1_test.npy'), np.array(test_data3_feats_label))




