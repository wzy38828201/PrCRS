import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from train_data2迁移 import Dataset, Collate
from model import tranMedical
#import matplotlib.pyplot as plt
#import pandas as pd
from statistics import mean 
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

torch.manual_seed(1234)

def validate1(model, data_loader, criterion, mode='train'):
    model.eval()
    y_pred_scores, y_preds = [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            #print(x)
            x = x.to(device)
            #y = y.cuda()
            y_pred = model(x)
            
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred_score = y_pred.cpu().numpy()[0][1]
            y_pred_label = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]
            # print('y_pred: ',  y_pred[0][1])
            #y_label = y.cpu().numpy()[0]
            y_pred_scores += [y_pred_score]
            y_preds += [y_pred_label]
            #y_labels += [y_label]
        
        return y_pred_scores, y_preds#, y_labels

def validate(model, data_loader, criterion, mode='train'):
    model.eval()
    y_pred_scores, y_preds, y_labels = [], [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred_score = y_pred.cpu().numpy()[0][1]
            y_pred_label = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]
            # print('y_pred: ',  y_pred[0][1])
            y_label = y.cpu().numpy()[0]
            y_pred_scores += [y_pred_score]
            y_preds += [y_pred_label]
            y_labels += [y_label]
        
        return y_pred_scores, y_preds, y_labels     

#xx = [[1,2,3,4,5,6,7,000]]#最后一个000是固定的，不用管它。医生输入的是前面的内容
xx = [[str(sys.argv[1]),000]]
        
#print(np.array(xx).shape[1])
#step1: define classifier
if np.array(xx).shape[1]==8:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w只细.xlsx'
    ckpt_path = './output多/只细胞因子,0.0003,24/97.pt'
elif np.array(xx).shape[1]==43:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w全.xlsx'
    ckpt_path = './output多/全部,0.0002,20/50.pt'
elif np.array(xx).shape[1]==36:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w除细.xlsx'
    ckpt_path = './output多/除去细胞因子,0.0003,20/18.pt'
elif np.array(xx).shape[1]==17:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w只生化.xlsx'
    ckpt_path = './output多/只生化项,0.0002,20/37.pt'
elif np.array(xx).shape[1]==11:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w只血常规.xlsx'
    ckpt_path = './output多/只血常规,0.00025,16/80.pt'
elif np.array(xx).shape[1]==27:
    xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\w生化血常规.xlsx'
    ckpt_path = './output多/生化血常规,0.0002,16/97.pt'

model = tranMedical(num_class=2).to(device)
ckpt = torch.load(ckpt_path)['model']
model.load_state_dict(ckpt)
# print('model: ', model)
model.eval()

collate_fn = Collate()
criterion = torch.nn.CrossEntropyLoss()

#这里的下面三个只能选一个去测试，不然的话会重复y_labels导致结果错误
#这里有w的原因是这里要测试

with torch.no_grad():
    np.save(os.path.join(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a', 'da.npy'), np.array(xx))
    #print('aaaaaaa')
    testset = Dataset(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\a\da.npy')
#    testset = Dataset(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\data2\data2_test多.npy')
    #print('aaaaaaa')
    test_loader = DataLoader(testset, num_workers=0,
                          shuffle=False, batch_size=1,
                          pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)
    y_pred_scores, y_preds = validate1(model, test_loader, criterion)
    

    thresh = 0.3

    y_preds = [int(x > thresh) for x in y_pred_scores]
    #print(f'y_pred_scores: {y_pred_scores}')
    #print(f'y_preds:       {y_preds}')
    
#    #输出对应的发生概率值前提文件
#    ddff = pd.DataFrame()
#    ddff[0] = y_pred_scores
#    ddff[1] = y_preds
#    ddff[2] = y_labels
#    ddff.to_excel(r'C:\Users\lenovo\Desktop\w2.xlsx')

#下面就是预测输出结果发生的概率的
df = pd.read_excel(xlsx_file2,header = None)

di = []
for i,j in enumerate(df[0]):
    for m,n in enumerate(df[2]):
        if i==m:
            if i<5:
                av = df[2][:m+5].tolist()
                di.append(mean(av))
            else:
                if j>=0.3:
                    av = df[2][m-5:m+3].tolist()
                    di.append(mean(av))
                else:
                    if j>=0.1:
                        av = df[2][m-10:m+3].tolist()
                        di.append(mean(av))
                    else:
                        av = df[2][m-15:m+5].tolist()
                        di.append(mean(av))
    
df2 = pd.DataFrame()
df2[0] = df[0]
di.sort(reverse=True)
df2[1] = di

#找出和某个值最相近的那个值
y = list(df2[0])
uu = []
for yp in y_pred_scores:
    ay = yp
for ii in y:
    uu.append(abs(ay-ii))
#y = random.randrange(1,100)
grades_abs = list(map(lambda x:x,uu))
peo_grade = list(np.where(np.array(grades_abs) == min(grades_abs))[0])
for pe in peo_grade:
    peo = pe
#print(df2[0][peo])
print(df2[1][peo])

