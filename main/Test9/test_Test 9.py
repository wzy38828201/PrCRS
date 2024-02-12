import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import tranMedical
import pandas as pd
from statistics import mean 

from sklearn.metrics import confusion_matrix,roc_curve, auc,recall_score, accuracy_score, precision_score, f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
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
    y_pred_scores, y_labels = [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            y_pred = F.softmax(y_pred, dim=-1)
#            logits += y_pred
            y_pred_score = y_pred.cpu().numpy()[0][1]
            #y_pred_label = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]
            y_label = y.cpu().numpy()[0]
            
            y_pred_scores += [y_pred_score]
            #y_preds += [y_pred_label]
            y_labels += [y_label]
        
        return y_pred_scores, y_labels      

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
    plt.savefig(r'C:\Users\lenovo\Desktop\roc.png')
    plt.show()
    return roc_auc



# step1: define classifier
#ckpt_path = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output多\比值,0.0003,20\6.pt'
#xx = [[1,2,3,4,5,6,7,000]]#最后一个000是固定的，不用管它。医生输入的是前面的内容
xx = [['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
            '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
            '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
            '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
             'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A',000]]

#print(np.array(xx).shape[1])
#step1: define classifier
if np.array(xx).shape[1]==43:
    ckpt_path = r'main/Test9/model/result.pt'#The results of "train_data2_test9" run save the model
    
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
    testset = Dataset('main/Test9/data193.npy')#'./data2/data2_test多.npy'
    test_loader = DataLoader(testset, num_workers=0,
                          shuffle=False, batch_size=1,
                          pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)
    y_pred_scores, y_labels = validate(model, test_loader, criterion)
    
    wz = pd.DataFrame()
    wz[0] = y_pred_scores
    wz[1] = y_labels
    wz[2] = range(len(y_pred_scores))
    wz = wz.sort_values(0,ascending=False)
df = wz

di = []
for i,j in enumerate(df[0]):
    for m,n in enumerate(df[1]):
        if i==m:
            if i<5:
                av = df[1][:m+5].tolist()
                di.append(mean(av))
            else:
                if j>=0.3:
                    av = df[1][m-5:m+3].tolist()
                    di.append(mean(av))
                else:
                    if j>=0.1:
                        av = df[1][m-10:m+5].tolist()
                        di.append(mean(av))
                    else:
                        av = df[1][m-15:m+10].tolist()
                        di.append(mean(av))
    
df2 = pd.DataFrame()
df2[0] = df[0]
di.sort(reverse=True)
df2[1] = di
df2[2] = df[2]
df2[3] = df[1]
df3 = df2.sort_values(2)
df4 = pd.DataFrame()
df4[0] = df3[0]
df4[1] = df3[1]
df4[2] = df3[3]
df4.to_excel(r'main/Test9/W193_all.xlsx',index = None)


testset = Dataset(r'main/Test9/data9.npy')#'./data2/data2_test多.npy'
test_loader = DataLoader(testset, num_workers=0,
                      shuffle=False, batch_size=1,
                      pin_memory=False, drop_last=True,
                      collate_fn=collate_fn)
y_pred_scores2, y_labels = validate(model, test_loader, criterion)
#找出和某个值最相近的那个值
y = list(df4[0])
shuchu = []
for yp in y_pred_scores2:
    ay = yp#取的是最后的那个值
    uu = []
    for ii in y:
        uu.append(abs(ay-ii))
    #y = random.randrange(1,100)
    grades_abs = list(map(lambda x:x,uu))
    peo_grade = list(np.where(np.array(grades_abs) == min(grades_abs))[0])
    for pe in peo_grade:
        peo = pe
    #print(df2[0][peo])
    shuchu.append(df4[1][peo])
gai = pd.DataFrame()
gai[0] = y_pred_scores2
gai[1] = shuchu
gai.to_excel(r'main/Test9/W9_all.xlsx',index = None)