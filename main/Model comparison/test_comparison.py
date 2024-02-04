import numpy as np
#import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import pandas as pd
import matplotlib.pyplot as plt
from train_data2迁移 import Dataset, Collate
from model import tranMedical
#import matplotlib.pyplot as plt
import pandas as pd
#from statistics import mean 
#import sys
from sklearn.metrics import confusion_matrix,roc_curve, auc,recall_score, accuracy_score, precision_score, f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.manual_seed(1234)

def validate(model, data_loader, criterion, mode='train'):
    model.eval()
    y_pred_scores, y_preds, y_labels, logits = [], [], [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            y_pred = F.softmax(y_pred, dim=-1)
            logits += y_pred
            y_pred_score = y_pred.cpu().numpy()[0][1]
            #y_pred_label = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]
            y_label = y.cpu().numpy()[0]
            
            y_pred_scores += [y_pred_score]
            #y_preds += [y_pred_label]
            y_labels += [y_label]
        
        return y_pred_scores, y_labels,logits        

def plot(y_pred, y_label):
    # y_label = ([1, 1, 1, 2, 2, 2])
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    fpr, tpr, thersholds = roc_curve(y_label, y_pred, pos_label=1)
    
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(r'C:\Users\lenovo\Desktop\roc.png')
    plt.show()
    return roc_auc


def main():
    xx = [['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
                '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
                '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
                '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
                 'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A',000]]
    
    #print(np.array(xx).shape[1])
    #step1: define classifier
    if np.array(xx).shape[1]==43:
        ckpt_path = 'main/Model comparison/PrCRS/all.pt'
    
    model = tranMedical(num_class=2).cuda()
    ckpt = torch.load(ckpt_path)['model']
    model.load_state_dict(ckpt)
    # print('model: ', model)
    model.eval()

    collate_fn = Collate()
    criterion = torch.nn.CrossEntropyLoss()

    # class2ids = {'negative': 0, 'positive': 1,}
    # id2classes = {class2ids[key]: key for key in class2ids}

    with torch.no_grad():
        testset = Dataset('main/data/1_day/data2_test2.npy')
        test_loader = DataLoader(testset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)
        y_pred_scores, y_labels,logits = validate(model, test_loader, criterion)
        #print(y_pred_scores, y_preds, y_labels)
        trainset = Dataset('main/data/1_day/data2_train2.npy')
        train_loader = DataLoader(trainset, num_workers=0,
                       shuffle=False, batch_size=1,
                       pin_memory=False, drop_last=True,
                       collate_fn=collate_fn)
        y_pred_scorestrain, y_preds, y_labels = validate(model, train_loader, criterion)
        
        valset = Dataset('main/data/1_day/data2_valid2.npy')
        val_loader = DataLoader(valset, num_workers=0,
                               shuffle=False, batch_size=1,
                               pin_memory=False, drop_last=True,
                               collate_fn=collate_fn)
        y_pred_scoresvalid, y_preds, y_labels = validate(model, val_loader, criterion)
        
        print(len(y_pred_scores),len(y_pred_scorestrain),len(y_pred_scoresvalid))
        zzz = y_pred_scores+y_pred_scorestrain+y_pred_scoresvalid
        print(len(zzz))

#        thresh = 0.3
#
#        y_preds = [int(x > thresh) for x in y_pred_scores]
#        print(f'y_pred_scores: {y_pred_scores}')
#        print(f'y_preds:       {y_preds}')
#        print(f'y_labels:      {y_labels}')
#        print(f'logits:      {logits}')
#
#
#        tn, fp, fn, tp = confusion_matrix(y_labels, y_preds).ravel()
#        auc = plot(y_preds, y_labels)#画图
#        acc = accuracy_score(y_labels, y_preds)
#        precision = precision_score(y_labels, y_preds, average='macro')
#        recall = recall_score(y_labels, y_preds, average='macro')
#        Specificity = tn / (tn + fp)
#        
#        macro_f1 = f1_score(y_labels, y_preds, average='macro')
#        
#        print('auc: ', auc)
#        print('acc: ', acc)
#        print('precision: ', precision)
#        print('recall: ', recall)
#        print('Specificity: ', Specificity)
#        print('macro_f1: ', macro_f1)
        
if __name__ == '__main__':
    main()