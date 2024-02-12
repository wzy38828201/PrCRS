import os
import numpy as np
import random

import pandas as pd
pd.set_option('display.notebook_repr_html',False)
# 读取xls（绝对路径）


xlsx_file2 = 'main/data/CAR-T.xlsx'
df2 = pd.read_excel(xlsx_file2)

b = df2['姓名'].unique().tolist()

#Misalignment (one day in advance)
c = pd.DataFrame()
a = df2
for jj,ii in enumerate(b):
    aa = a[a['姓名'] == ii]
    di = pd.DataFrame()
    crs = list(aa['CRS'])
    del(crs[0])
    aa.index = range(len(aa))
    aa.drop([len(aa)-1],inplace=True)
    di = aa
    di['CRS'] = crs
    if jj==0:
        c = di
    if jj!=0:
        c = pd.concat([c,di],ignore_index=True)

df20 = c
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

data2=df2.values
keys2 = list(df2.keys())
print(keys2)
print(data2)

common_keys = keys2[5:-4]
##At least five more to identify
common_keys = ['D-dimer', 'Procalcitonin', 'Type B natriuretic peptide', 'Alpha hydroxybutyrate dehydrogenase', 'prealbumin','Tumor burde',
    'Plasma prothrombin time', 'Activated partial prothrombin time','fibrinogen', #Clotting term
    'red blood cell', 'hemoglobin', 'leukocyte','Neutrophil percentage', 'Neutrophil count', 'Lymphocyte percentage', 'Lymphocyte count','platelet', 'Monocyte percentage', 'Monocyte count', #Blood routine examination3
    'sodium', 'potassium', 'chlorine', 'calcium', 'Uric acid', 'glucose', 'triglyceride', 'gamma glutamyl transpeptidase','albumin', 'Glutamic pyruvic transaminase', 'Glutamic oxalacetic transaminase', 'Alkaline phosphatase', 'Lactate dehydrogenase', 'creatinine', 'C-reactive protein', 'ferritin',#Biochemical term2
     'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A','CRS'] #cytokine☆√1

print('common_keys: ', common_keys)

# print(df2['IFNγ'].values)


data1_label2id = {'Severe patients': 0, 'Mild patients':1, 'Follow-up patients': 2, 'Healthy controls': 3}
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


test_num = 30
savedir = 'main/data/1_day'

data2_feats = get_feats(df2, common_keys)
data2_label = [data2_label2id[x] for x in df2['CRS']]
print(data2_label)
data2_feats_label = [list(x) + [y] for x, y in zip(data2_feats, data2_label)]
random.shuffle(data2_feats_label)
train_data2_feats_label = data2_feats_label[int(0.4*len(data2_feats_label)):]#-2*test_num
valid_data2_feats_label = data2_feats_label[int(0.2*len(data2_feats_label)):int(0.4*len(data2_feats_label))]#-2*test_num:-test_num
test_data2_feats_label = data2_feats_label[:int(0.2*len(data2_feats_label))]#-test_num:

print('train_data2_feats_label: ', np.array(train_data2_feats_label).shape)
np.save(os.path.join(savedir, 'data2_train2.npy'), np.array(train_data2_feats_label))
np.save(os.path.join(savedir, 'data2_valid2.npy'), np.array(valid_data2_feats_label))
np.save(os.path.join(savedir, 'data2_test2.npy'), np.array(test_data2_feats_label))




