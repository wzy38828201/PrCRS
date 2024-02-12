import os
import numpy as np
import random

import pandas as pd
pd.set_option('display.notebook_repr_html',False)

df3_aval = pd.read_excel('main/data/COVID-19.xlsx')
keys3 = df3_aval.columns.tolist()
keys3.remove('性别')
aval_labels = list(df3_aval['最大严重性'])

xlsx_file2 = 'main/data/CAR-T.xlsx'
df2 = pd.read_excel(xlsx_file2)
data2=df2.values
keys2 = list(df2.keys())

#Compare which features are the same as the data to be predicted
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
savedir = 'main/data'
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




