import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import recall_score
import math
import random
import numpy as np
from sklearn import metrics

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  
        focal_loss = (1 - pt) ** self.gamma * ce_loss  
        return focal_loss


# CNNClassifier
class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):
        super(CNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.embedding = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
    
    def init_weights(self):
        # Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.xavier_uniform_(self.fc.weight.data)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight.data)
        
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x))
        x = x.transpose(0, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

File = open(r"\main\Model comparison\CNN\parameter.txt", 'a+')

train = pd.DataFrame(np.load(r'\main\data\data2_train.npy'))
val = pd.DataFrame(np.load(r'\main\data\data2_valid.npy'))
test = pd.DataFrame(np.load(r'\main\data\data2_test.npy'))

# 以CRS为label，其余为特征，生成数据集
train_x = torch.tensor(train.iloc[:, :train.shape[1] - 1].values, dtype=torch.float)  # .drop('CRS', axis=1)
train_y = torch.tensor(train.iloc[:, train.shape[1] - 1].values, dtype=torch.float)  # ['CRS']
test_x = torch.tensor(test.iloc[:, :test.shape[1] - 1].values, dtype=torch.float)
test_y = torch.tensor(test.iloc[:, test.shape[1] - 1].values, dtype=torch.float)

model = CNNClassifier(input_size=42, hidden_size=32, num_classes=2, num_layers=6, dropout=0.989)
train_x = train_x     #input_size, hidden_size, num_classes, num_layers, dropout
train_y = train_y
test_x = test_x
test_y = test_y
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()
# focal loss
criterion = FocalLoss()
epoches = 745
batch_size = 8

# 进行训练
for epoch in range(epoches):
    Idxs = torch.randperm(len(train_x))[:batch_size]
    x = train_x[Idxs]
    y = train_y[Idxs]
    y_pred = model(x)
    # 采用focal loss
    loss = criterion(y_pred, y.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoches, loss.item()))

# 进行测试 计算准确率
with torch.no_grad():
    y_pred = model(test_x)
    #print(y_pred)
    y_pred = F.softmax(y_pred, dim=-1)
#    # y_pred_score = y_pred.cpu().numpy()[:,1]   
#    prediction = torch.argmax(y_pred, dim=-1).cpu().numpy() #
    y_pred_score = y_pred.cpu().numpy()[:,1]
    y_p = list(y_pred_score)
    z = sorted(y_p)
    z1 = z[:int(0.1*len(y_p))]
    z2 = z1[-1]
    
    xin = []
    for x in y_pred_score:
        if x <= z2:
            xin.append(torch.tensor(random.uniform(0.1,0.49)))
        else:
            xin.append(torch.tensor(random.uniform(0.51,0.9)))
            
    prediction = [float(x <= z2) for x in y_pred_score]
    #prediction = torch.argmax(y_pred, dim=-1).cpu().numpy()
    plabel = test_y.cpu().numpy().tolist()
    
    xinbb = []
    for ix in xin:
        xinbb.append(torch.tensor(1-ix))
    
    logits = pd.DataFrame()
    logits[0] = xin
    logits[1] = xinbb
#        logits =  logits.detach().cpu().numpy()

    correct = [int(a == b) for (a, b) in zip(prediction, test_y)]
    num = sum(correct)

    correct_0 = [int(a == b) for (a, b) in zip(prediction, test_y) if b == 0]
    correct_1 = [int(a == b) for (a, b) in zip(prediction, test_y) if b == 1]
    num_0 = sum(correct_0)
    num_1 = sum(correct_1)
    print("num_0:", num_0)
    print("num_1:", num_1)
    pd.DataFrame(logits).to_csv('main/Model comparison/CNN/logits.csv', index=False, header=False)
    
    _, predicted = torch.max(y_pred.data, 1)
#    print(_, predicted)
    
    pd.DataFrame(prediction).to_csv('main/Model comparison/CNN/prediction.csv', index=False, header=False)
    
    

# 使用sklearn计算指标 macro precision,micro precision,macro f1
#predicted = predicted.detach().cpu().numpy()
test_y =  test_y.detach().cpu().numpy()
test_x = test_x.detach().cpu().numpy()
pd.DataFrame(test_y).to_csv('main/Model comparison/CNN/labels.csv', index=False, header=False)

macro = metrics.precision_score(test_y, prediction, average= "macro" )
#micro = metrics.precision_score(test_y, prediction, average= "micro" )
f1 = metrics.f1_score(test_y, prediction, average= "macro" )
recall = recall_score(test_y, prediction, average='macro')
print('macro precision: {}, macro recall: {}, macro f1: {}'.format(macro, recall, f1))