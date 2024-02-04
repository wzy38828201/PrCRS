import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

from model import tranMedical
from sklearn.metrics import roc_curve
#import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import recall_score, accuracy_score, precision_score
#import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from statistics import mean 
import sys

torch.manual_seed(1234)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.dataset = np.load(dataset_file, allow_pickle=True)
        print('self.dataset: ', self.dataset.shape)

    def __getitem__(self, index):
        data = self.dataset[index]
        
        feat, label = data[:-1], data[-1]
        if np.isin(np.nan, feat):
            print('label: ', label)
            print('feats: ', np.isin(np.nan, feat), feat)
        feat = torch.from_numpy(feat)
        label = torch.LongTensor([int(label)])

        return feat, label

    def __len__(self):
        return len(self.dataset)

class Collate(object):
    def __init__(self,):
        return

    def __call__(self, batch):
        B = len(batch)
        feat_size = batch[0][0].shape[0]
        feats = torch.FloatTensor(B, feat_size)
        labels = torch.LongTensor(B)
        for i in range(B):
            feats[i] = batch[i][0]
            labels[i] = batch[i][1]

        return feats, labels


def validate1(model, data_loader, criterion, mode='train'):
    model.eval()
    y_pred_scores, y_preds = [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            #print(x)
            x = x.cuda()
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
            x = x.cuda()
            y = y.cuda()
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

xx = [[str(sys.argv[1]),000]]#
#b = [[18.8,5.5,38.6,24.5,28,57,7.75,18.8,5.5,38.6,24.5,28,57,7.75,18.8,5.5,38.6,24.5,28,57,7.75,18.8,5.5,38.6,24.5,28,57,7.75,18.8,5.5,38.6,24.5,28,57,7.75,000]]
#step1: define classifier
if np.array(xx).shape[1]==8:
    xlsx_file2 = 'main/output2/2_day/cytokines.xlsx'
    ckpt_path = 'main/output2/2_day/cytokines.pt'
elif np.array(xx).shape[1]==43:
    xlsx_file2 = 'main/output2/2_day/all.xlsx'
    ckpt_path = 'main/output2/2_day/all.pt'
elif np.array(xx).shape[1]==36:
    xlsx_file2 = 'main/output2/2_day/decytokine.xlsx'
    ckpt_path = 'main/output2/2_day/decytokine.pt'
elif np.array(xx).shape[1]==17:
    xlsx_file2 = 'main/output2/2_day/Biochemical_term.xlsx'
    ckpt_path = 'main/output2/2_day/Biochemical_term.pt'
elif np.array(xx).shape[1]==11:
    xlsx_file2 = 'main/output2/2_day/blood_routine.xlsx'
    ckpt_path = 'main/output2/2_day/blood_routine.pt'
elif np.array(xx).shape[1]==27:
    xlsx_file2 = 'main/output2/2_day/Biochemical_blood_routine.xlsx'
    ckpt_path = 'main/output2/2_day/Biochemical_blood_routine.pt'

model = tranMedical(num_class=2).cuda()
ckpt = torch.load(ckpt_path)['model']
model.load_state_dict(ckpt)
# print('model: ', model)
model.eval()

collate_fn = Collate()
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    np.save(os.path.join('main/output2/2_day', 'db.npy'), np.array(xx))
    #print('aaaaaaa')
    testset = Dataset('main/output2/2_day/db.npy')
    #print('aaaaaaa')
    test_loader = DataLoader(testset, num_workers=0,
                          shuffle=False, batch_size=1,
                          pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)
    y_pred_scores, y_preds= validate1(model, test_loader, criterion)
    

    thresh = 0.3

    y_preds = [int(x > thresh) for x in y_pred_scores]
    #print(f'y_pred_scores: {y_pred_scores}')
    #print(f'y_preds:       {y_preds}')
    
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

#Find the value that is closest to the value
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

