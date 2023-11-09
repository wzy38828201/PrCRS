import os
import numpy as np
import random

import pandas as pd
pd.set_option('display.notebook_repr_html',False)
# 读取xls（绝对路径）

xlsx_file1 = './data/data1.xlsx'
df1 = pd.read_excel(xlsx_file1)
keys1 = df1.keys()
keys1 = [key.replace('-', '') for key in keys1]
df1.columns = keys1
data1=df1.values
print(keys1)
print(data1)

xlsx_file2 = './data/data2.xlsx'
df2 = pd.read_excel(xlsx_file2)
data2=df2.values
keys2 = list(df2.keys())
print(keys2)
print(data2)

common_keys = [key for key in keys1 if key in keys2]
print('common_keys: ', common_keys)

# print(df2['IFNγ'].values)


data1_label2id = {'Severe patients': 0, 'Mild patients':1, 'Follow-up patients': 2, 'Healthy controls': 3}
data2_label2id = {0: 0, 1: 0, 2:0, 3:1, 4:1}

def get_feats(df, common_keys):
    tem = []
    for key in common_keys:
        tem.append(df[key].values)
    
    tem = np.array(tem).T

    feats = [x for x in tem]
    # feats = [(x-np.min(x))/(np.max(x)-np.min(x)) for x in feats]
    feats = [(x-np.mean(x))/(np.std(x) + 1e-4) for x in feats]

    return feats


test_num = 10
savedir = './data'

data1_feats = get_feats(df1, common_keys)
data1_label = [data1_label2id[x] for x in df1['Type']]
data1_feats_label = [list(x) + [y] for x, y in zip(data1_feats, data1_label)]
random.shuffle(data1_feats_label)
train_data1_feats_label = data1_feats_label[:-2*test_num]
valid_data1_feats_label = data1_feats_label[-2*test_num:-test_num]
test_data1_feats_label = data1_feats_label[-test_num:]

print('train_data1_feats_label: ', np.array(train_data1_feats_label).shape)
np.save(os.path.join(savedir, 'data1_train.npy'), np.array(train_data1_feats_label))
np.save(os.path.join(savedir, 'data1_valid.npy'), np.array(valid_data1_feats_label))
np.save(os.path.join(savedir, 'data1_test.npy'), np.array(test_data1_feats_label))

data2_feats = get_feats(df2, common_keys)
data2_label = [data2_label2id[x] for x in df2['CRS']] 
data1_label = [data1_label2id[x] for x in df1['Type']]
data2_feats_label = [list(x) + [y] for x, y in zip(data2_feats, data2_label)]
random.shuffle(data2_feats_label)
train_data2_feats_label = data2_feats_label[:-2*test_num]
valid_data2_feats_label = data2_feats_label[-2*test_num:-test_num]
test_data2_feats_label = data2_feats_label[-test_num:]

print('train_data2_feats_label: ', np.array(train_data2_feats_label).shape)
np.save(os.path.join(savedir, 'data2_train.npy'), np.array(train_data2_feats_label))
np.save(os.path.join(savedir, 'data2_valid.npy'), np.array(valid_data2_feats_label))
np.save(os.path.join(savedir, 'data2_test.npy'), np.array(test_data2_feats_label))




