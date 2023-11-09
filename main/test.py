import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train_data2_本身 import Dataset, Collate, validate
from model import tranMedical
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
from sklearn.metrics import confusion_matrix
torch.manual_seed(1234)

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

            
            

def plot(y_pred, y_label):
    # y_label = ([1, 1, 1, 2, 2, 2])
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    fpr, tpr, thersholds = roc_curve(y_label, y_pred, pos_label=1)
    
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc.png')
    return roc_auc


def main():
    # step1: define classifier
    ckpt_path = './outputs2/19.pt'
    model = tranMedical(num_class=2).cuda()
    ckpt = torch.load(ckpt_path)['model']
    model.load_state_dict(ckpt)
    # print('model: ', model)
    model.eval()

    collate_fn = Collate()
    criterion = torch.nn.CrossEntropyLoss()

    class2ids = {'negative': 0, 'positive': 1,}
    id2classes = {class2ids[key]: key for key in class2ids}

    #这里有w的原因是这里要测试
    with torch.no_grad():
        # testset = Dataset('./data2/data2_test.npy')
        # test_loader = DataLoader(testset, num_workers=0,
        #                       shuffle=False, batch_size=1,
        #                       pin_memory=False, drop_last=True,
        #                       collate_fn=collate_fn)
        # y_pred_scores, y_preds, y_labels = validate(model, test_loader, criterion)

        # trainset = Dataset('./data2/data2_train.npy')
        # train_loader = DataLoader(trainset, num_workers=0,
        #                       shuffle=False, batch_size=1,
        #                       pin_memory=False, drop_last=True,
        #                       collate_fn=collate_fn)
        # y_pred_scores, y_preds, y_labels = validate(model, train_loader, criterion)
        #
        valset = Dataset('./data2/data2_valid.npy')
        val_loader = DataLoader(valset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)
        y_pred_scores, y_preds, y_labels = validate(model, val_loader, criterion)

        # zzz = y_pred_scores+y_pred_scorestrain+y_pred_scoresvalid
        # pd.DataFrame(zzz).to_excel(r'G:\CRS预测\程序分析\分类预测\决策树可视化\xgb_visualization\批量生成\对应概率图\概率.xlsx')
        thresh = 0.3

        y_preds = [int(x > thresh) for x in y_pred_scores]
        print(f'y_pred_scores: {y_pred_scores}')
        print(f'y_preds:       {y_preds}')
        print(f'y_labels:      {y_labels}')

        tn, fp, fn, tp = confusion_matrix(y_labels, y_preds).ravel()
        auc = plot(y_preds, y_labels)
        acc = accuracy_score(y_labels, y_preds)
        precision = precision_score(y_labels, y_preds)
        recall = recall_score(y_labels, y_preds)
        Specificity = tn / (tn + fp)

        print('auc: ', auc)
        print('acc: ', acc)
        print('precision: ', precision)
        print('recall: ', recall)
        print('Specificity: ', Specificity)

if __name__ == '__main__':
    main()