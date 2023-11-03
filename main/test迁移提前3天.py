import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train_data2迁移 import Dataset, Collate, validate
from model import tranMedical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
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
    
    plt.plot(fpr, tpr, 'k--', color='red', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
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
#    xx = [['红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
#                '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',000]]    #step1: define classifier
#    xx = [['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
#        '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
#        '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
#        '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',#生化项2
#         'IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A',000]] #细胞因子☆√1
#    xx = [['D-二聚体', '降钙素原', 'B型尿钠肽', 'α羟丁酸脱氢酶', '前白蛋白','原幼细胞',#和下面的凝血项一起为2
#        '血浆凝血酶原时间', '活化部分凝血酶原时间','纤维蛋白原', #凝血项
#        '红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
#        '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',000]]#生化项2
#    xx = [['IL2', 'IL4', 'IL6', 'IL10', 'TNFα', 'IFNγ', 'IL17A',000]]
#    xx = [['钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',000]]
#    xx = [['红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数',000]]
    xx = [['红细胞', '血红蛋白', '白细胞','中性粒细胞百分比', '中性粒细胞计数', '淋巴细胞百分比', '淋巴细胞计数','血小板', '单核细胞百分比', '单核细胞计数', #血常规3
           '钠', '钾', '氯', '钙', '尿酸', '葡萄糖', '甘油三酯', 'γ-谷氨酰转肽酶','白蛋白', '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '乳酸脱氢酶', '肌酐', 'C反应蛋白', '铁蛋白',000]]

    if np.array(xx).shape[1]==8:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w只细.xlsx'
        ckpt_path = './output迁移提前3天/只细胞因子,0.0002,18/4.pt'
    elif np.array(xx).shape[1]==43:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w全.xlsx'
        ckpt_path = './output迁移提前3天/全部1,0.0002,24/49.pt'
    elif np.array(xx).shape[1]==36:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w除细.xlsx'
        ckpt_path = './output迁移提前3天/除去细胞因子,0.0003,18/87.pt'
    elif np.array(xx).shape[1]==17:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w只生化项.xlsx'
        ckpt_path = './output迁移提前3天/只生化项,0.00035,16/34.pt'
    elif np.array(xx).shape[1]==11:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w只血常规.xlsx'
        ckpt_path = './output迁移提前3天/只血常规,0.00025,18/73.pt'
    elif np.array(xx).shape[1]==27:
        xlsx_file2 = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\output迁移提前3天\c\w生化血常规.xlsx'
        ckpt_path = './output迁移提前3天/生化血常规,0.00025,20/69.pt'
    
    model = tranMedical(num_class=2).cuda()
    ckpt = torch.load(ckpt_path)['model']
    model.load_state_dict(ckpt)
    # print('model: ', model)
    model.eval()

    collate_fn = Collate()
    criterion = torch.nn.CrossEntropyLoss()

    # class2ids = {'negative': 0, 'positive': 1,}
    # id2classes = {class2ids[key]: key for key in class2ids}

    #这里的下面三个只能选一个去测试，不然的话会重复y_labels导致结果错误
    #这里有w的原因是这里要测试
    with torch.no_grad():
        testset = Dataset('./data提前3天/data2_test多2.npy')
        test_loader = DataLoader(testset, num_workers=0,
                               shuffle=False, batch_size=1,
                               pin_memory=False, drop_last=True,
                               collate_fn=collate_fn)
        y_pred_scores, y_preds, y_labels = validate(model, test_loader, criterion)

        # trainset = Dataset('./data提前2天/data2_train多2.npy')
        # train_loader = DataLoader(trainset, num_workers=0,
        #                       shuffle=False, batch_size=1,
        #                       pin_memory=False, drop_last=True,
        #                       collate_fn=collate_fn)
        # y_pred_scores, y_preds, y_labels = validate(model, train_loader, criterion)
        #
#        valset = Dataset('./data提前2天/data2_valid多2.npy')
#        val_loader = DataLoader(valset, num_workers=0,
#                              shuffle=False, batch_size=1,
#                              pin_memory=False, drop_last=True,
#                              collate_fn=collate_fn)
#        y_pred_scores, y_preds, y_labels = validate(model, val_loader, criterion)

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