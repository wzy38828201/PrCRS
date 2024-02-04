
import numpy as np
import pandas as pd

def prob(paths):
    for path in paths:
        prob = []
        with open(path + '\logits.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                a = []
                s = line.split(',')
                #p0 = float(s[0].split('(')[1])
                p0 = s[0].split('(')[1]
                p0 = float(p0.split(')')[0])
                #print(p0)
                
                #p1 = float(s[0].split('(')[1])
                p1 = s[1].split('(')[1]
                p1 = float(p1.split(')')[0])
                #print(p1)
                # p3 = float(s[6].split('(')[1])  # DBLP
                a.append(p0)
                a.append(p1)
#                print(p0,p1)
                #a.append(p2)
                # a.append(p3)  # DBLP
                arr = np.array(a)
#                print(arr)
                softmax_z = np.exp(arr) / sum(np.exp(arr))
#                print(np.exp(arr),sum(np.exp(arr)),softmax_z)
                prob.append(softmax_z)

        pd.DataFrame(prob).to_excel(path + '\prob.xlsx', index=False, header=['0', '1'])
        # pd.DataFrame(prob).to_csv(path + 'prob.csv',index=False,header=['0','1','2','3'])  # DBLP

def label(paths):
    for path in paths:
        labels = []

        with open(path + '\labels.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                la = float(line.replace('\n',''))
                if la == 0:
                    labels.append((1,0))
                elif la == 1:
                    labels.append((0,1))
        pd.DataFrame(np.array(labels)).to_excel(path + '\labels_2.xlsx', index=False,header=False)



paths = [
    #'main/Model comparison/PrCRS',
    #'main/Model comparison/Squeezeformer'
    #'main/draw/CNN'
    'main/draw/Transformer'
         ]

prob(paths)
label(paths)
