# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

n = ['D-dimer', 'Procalcitonin', 'Type B natriuretic peptide', 'Alpha hydroxybutyrate dehydrogenase', 'prealbumin','Tumor burde',
    'Plasma prothrombin time', 'Activated partial prothrombin time','fibrinogen', #Clotting term
    'red blood cell', 'hemoglobin', 'leukocyte','Neutrophil percentage', 'Neutrophil count', 'Lymphocyte percentage', 'Lymphocyte count','platelet', 'Monocyte percentage', 'Monocyte count', #Blood routine examination3
    'sodium', 'potassium', 'chlorine', 'calcium', 'Uric acid', 'glucose', 'triglyceride', 'gamma glutamyl transpeptidase','albumin', 'Glutamic pyruvic transaminase', 'Glutamic oxalacetic transaminase', 'Alkaline phosphatase', 'Lactate dehydrogenase', 'creatinine', 'C-reactive protein', 'ferritin',#Biochemical term2
     'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A','CRS'] #cytokine☆√1

a = pd.read_excel('main/Test9/9.xlsx')

load_npy = np.load("main/Test9/data9.npy")

load_npy = pd.DataFrame(load_npy)

load_npy.columns = n

shu = pd.DataFrame(np.arange(51).reshape(1, 51))
shu.columns = a.columns
for i,j in enumerate(load_npy['Neutrophil percentage']):
    for m,n in enumerate(load_npy['Lymphocyte percentage']):
        if m==i:
            for m1,n1 in enumerate(load_npy['Monocyte percentage']):
                if m1==i:
                    for m2,n2 in enumerate(load_npy['leukocyte']):
                        if m2==i:
                            for m3,n3 in enumerate(load_npy['IL6']):
                                if m3==i:
                                    for m4,n4 in enumerate(load_npy['Procalcitonin']):
                                        if m4==i:
                                            z1 = a[a['Neutrophil percentage'] == j]
                                            z2 = z1[z1['Lymphocyte percentage'] == n]
                                            z3 = z2[z2['Monocyte percentage'] == n1]
                                            z4 = z3[z3['leukocyte'] == n2]
                                            z5 = z4[z4['IL6'] == n3]
                                            z6 = z5[z5['Procalcitonin'] == n4]
                                            shu = pd.concat([shu,z6], axis=0)                            
shu = shu[1:].drop_duplicates()

z = shu
z.index = list(range(len(z)))
b = pd.read_excel('main/Test9/W9_all.xlsx')
z[0] = b[0]
z[1] = b[1]
z.to_excel('main/data/Actual probability/Probabilistic_correspondence.xlsx',index = 0)
