
import numpy as np
import pandas as pd

'''处理保存的labels.csv和logits.csv文件，用来画ROC曲线'''
def prob(paths):
    for path in paths:
        prob = []
        # path  = '20220912最新结果/原始/ACM/'
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
        # path  = '20220912最新结果/原始/ACM/'
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
    #r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\有迁移',
    #r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\无迁移'
    #r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\CNN'
    #r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\LSTM'
    r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\Transformer'
         ]

prob(paths)
label(paths)
