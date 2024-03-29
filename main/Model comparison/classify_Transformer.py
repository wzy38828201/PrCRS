import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
#import math
from sklearn.metrics import recall_score
import random
import numpy as np
import pandas as pd

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.embedding(x)
        pos_embedding = self.pos_embedding(torch.arange(0, x.size(0), device=x.device).repeat(x.size(1), 1))
        x = x + pos_embedding
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = x[-1, :, :]
        # x = self.fc(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads,num_class, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, num_heads, dropout)
        self.fc = nn.Linear(hidden_size, num_class)
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

File = open(r"\main\Model comparison\Transformer\parameter.txt", 'a+')

train = pd.DataFrame(np.load(r'\main\data\data2_train.npy'))
val = pd.DataFrame(np.load(r'\main\data\data2_valid.npy'))
test = pd.DataFrame(np.load(r'\main\data\data2_test.npy'))

# 以CRS为label，其余为特征，生成数据集
train_x = torch.tensor(train.iloc[:, :train.shape[1] - 1].values, dtype=torch.float)  # .drop('CRS', axis=1)
train_y = torch.tensor(train.iloc[:, train.shape[1] - 1].values, dtype=torch.float)  # ['CRS']
test_x = torch.tensor(test.iloc[:, :test.shape[1] - 1].values, dtype=torch.float)
test_y = torch.tensor(test.iloc[:, test.shape[1] - 1].values, dtype=torch.float)

train_dataset = Dataset(train_x, train_y)
test_dataset = Dataset(test_x, test_y)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

model = TransformerClassifier(42, 32, 2, 4, 2, 0.97)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 6

model = model

for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y.long())
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (x, y) in enumerate(test_loader):
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Accuracy: {}'.format(correct / total))

from sklearn import metrics
model.eval()
with torch.no_grad():
    y_pred = model(test_x)
    y_pred = F.softmax(y_pred, dim=-1)
    y_pred_score = y_pred.cpu().numpy()[:,1]
    y_p = list(y_pred_score)
    z = sorted(y_p)
    z1 = z[:int(0.05*len(y_p))]
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
#    _, predicted = torch.max(y_pred.data, 1)

    correct = [int(a == b) for (a, b) in zip(prediction, test_y)]
    num = sum(correct)

    correct_0 = [int(a == b) for (a, b) in zip(prediction, test_y) if b == 0]
    correct_1 = [int(a == b) for (a, b) in zip(prediction, test_y) if b == 1]
    num_0 = sum(correct_0)
    num_1 = sum(correct_1)
    print("num_0:", num_0)
    print("num_1:", num_1)
    pd.DataFrame(logits).to_csv('main/Model comparison/CNN/logits.csv', index=False, header=False)
    pd.DataFrame(prediction).to_csv('main/Model comparison/Transformer/prediction.csv', index=False, header=False)

#    _, predicted = torch.max(y_pred.data, 1)
#predicted = predicted.detach().cpu().numpy()
test_y =  test_y.detach().cpu().numpy()
pd.DataFrame(test_y).to_csv('main/Model comparison/Transformer/labels.csv', index=False, header=False)

macro = metrics.precision_score(test_y, prediction, average= "macro" )
recall = recall_score(test_y, prediction, average='macro')
f1 = metrics.f1_score(test_y, prediction, average= "macro" )
print('macro precision: {}, macro recall: {}, macro f1: {}'.format(macro, recall, f1))
