import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import tranMedical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
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
    
    plt.plot(fpr, tpr, 'k--', color='red', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontproperties = 'Times New Roman', size = 30)
    plt.yticks(fontproperties = 'Times New Roman', size = 30)
    plt.xlabel('False Positive Rate',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)
    plt.ylabel('True Positive Rate',fontdict={'family' : 'Times New Roman', 'size':30},labelpad=5)  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve',fontdict={'family' : 'Times New Roman', 'size':30})
    font = fm.FontProperties(family="Times New Roman", size=18)
    plt.legend(loc="lower right",prop=font)
    plt.savefig(r'C:\Users\lenovo\Desktop\roc.png',dpi = 300,bbox_inches='tight')
    plt.show()
    return roc_auc

def main():
    #xx = [['D-dimer', 'Procalcitonin', 'Type B natriuretic peptide', 'Alpha hydroxybutyrate dehydrogenase', 'prealbumin','Tumor burde',
    #    'Plasma prothrombin time', 'Activated partial prothrombin time','fibrinogen', #Clotting term
    #    'red blood cell', 'hemoglobin', 'leukocyte','Neutrophil percentage', 'Neutrophil count', 'Lymphocyte percentage', 'Lymphocyte count','platelet', 'Monocyte percentage', 'Monocyte count', #Blood routine examination3
    #    'sodium', 'potassium', 'chlorine', 'calcium', 'Uric acid', 'glucose', 'triglyceride', 'gamma glutamyl transpeptidase','albumin', 'Glutamic pyruvic transaminase', 'Glutamic oxalacetic transaminase', 'Alkaline phosphatase', 'Lactate dehydrogenase', 'creatinine', 'C-reactive protein', 'ferritin',#Biochemical term2
    #     'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A','CRS',000]] #cytokine☆√1

    if np.array(xx).shape[1]==8:
        xlsx_file2 = 'main/output2/3_day/cytokines.xlsx'
        ckpt_path = 'main/output2/3_day/cytokines.pt'
    elif np.array(xx).shape[1]==43:
        xlsx_file2 = 'main/output2/3_day/all.xlsx'
        ckpt_path = 'main/output2/3_day/all.pt'
    elif np.array(xx).shape[1]==36:
        xlsx_file2 = 'main/output2/3_day/decytokine.xlsx'
        ckpt_path = 'main/output2/3_day/decytokine.pt'
    elif np.array(xx).shape[1]==17:
        xlsx_file2 = 'main/output2/3_day/Biochemical_term.xlsx'
        ckpt_path = 'main/output2/3_day/Biochemical_term.pt'
    elif np.array(xx).shape[1]==11:
        xlsx_file2 = 'main/output2/3_day/blood_routine.xlsx'
        ckpt_path = 'main/output2/3_day/blood_routine.pt'
    elif np.array(xx).shape[1]==27:
        xlsx_file2 = 'main/output2/3_day/Biochemical_blood_routine.xlsx'
        ckpt_path = 'main/output2/3_day/Biochemical_blood_routine.pt'
    
    model = tranMedical(num_class=2).cuda()
    ckpt = torch.load(ckpt_path)['model']
    model.load_state_dict(ckpt)
    # print('model: ', model)
    model.eval()

    collate_fn = Collate()
    criterion = torch.nn.CrossEntropyLoss()

    # class2ids = {'negative': 0, 'positive': 1,}
    # id2classes = {class2ids[key]: key for key in class2ids}

    #Only one of the following three should be tested, otherwise y_labels will be repeated and the result will be incorrect
    with torch.no_grad():
        testset = Dataset('main/data/3_day/data2_test2.npy')
        test_loader = DataLoader(testset, num_workers=0,
                               shuffle=False, batch_size=1,
                               pin_memory=False, drop_last=True,
                               collate_fn=collate_fn)
        y_pred_scores, y_preds, y_labels = validate(model, test_loader, criterion)

        # trainset = Dataset('main/data/3_day/data2_train2.npy')
        # train_loader = DataLoader(trainset, num_workers=0,
        #                       shuffle=False, batch_size=1,
        #                       pin_memory=False, drop_last=True,
        #                       collate_fn=collate_fn)
        # y_pred_scores, y_preds, y_labels = validate(model, train_loader, criterion)
        #
#        valset = Dataset('main/data/3_day/data2_valid2.npy')
#        val_loader = DataLoader(valset, num_workers=0,
#                              shuffle=False, batch_size=1,
#                              pin_memory=False, drop_last=True,
#                              collate_fn=collate_fn)
#        y_pred_scores, y_preds, y_labels = validate(model, val_loader, criterion)

        # zzz = y_pred_scores+y_pred_scorestrain+y_pred_scoresvalid
        #pd.DataFrame(zzz).to_excel(r'main/output2/Chance.xlsx')
        
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