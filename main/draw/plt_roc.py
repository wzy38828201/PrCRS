import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from matplotlib import rcParams
'''画ROC曲线，计算AUC面积'''

#def micro(path):
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    # 计算每一类的ROC
#    y_test = pd.read_excel(path + 'labels_2.xlsx', header=None)
#    y_test = np.array(y_test)
#    y_score = pd.read_excel(path + 'prob.xlsx')
#    y_score = y_score.values
#    n_classes = 2
#    for i in range(2):
#        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    # Compute micro-average ROC curve and ROC area
#    ooo = y_test.ravel()
#    ooo1 = y_score.ravel()
#    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#    return fpr,tpr,roc_auc

def macro(path):
    y_test = pd.read_excel(path + 'labels_2.xlsx', header=None)
    y_test = np.array(y_test)
    y_score = pd.read_excel(path + 'prob.xlsx')
    y_score = y_score.values
    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr,tpr,roc_auc

def get_macro_acm_roc():
    att_fpr, att_tpr, att_roc_auc = macro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\无迁移\\')
    ori_fpr, ori_tpr, ori_roc_auc = macro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\有迁移\\')
    cnn_fpr, cnn_tpr, cnn_roc_auc = macro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\CNN\\')
    #lstm_fpr, lstm_tpr, lstm_roc_auc = macro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\LSTM\\')
    Transformer_fpr, Transformer_tpr, Transformer_roc_auc = macro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\Transformer\\')

    # Plot all ROC curves
    config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size":30
    #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)
    
    lw=2
    plt.figure(figsize=(10,8))
    plt.plot(cnn_fpr["macro"], cnn_tpr["macro"],
             label='CNN (AUC = {0:0.2f})'
                   ''.format(cnn_roc_auc["macro"]),
             color='red',
             linestyle='--',
             lw=lw)
        
    plt.plot(Transformer_fpr["macro"], Transformer_tpr["macro"],
             label='Transformer (AUC = {0:0.2f})'
                   ''.format(Transformer_roc_auc["macro"]),
             color='green',
             linestyle='--',
             lw=lw)
    
    plt.plot(att_fpr["macro"], att_tpr["macro"],
             label='Squeezeformer (AUC = {0:0.2f})'
                   ''.format(att_roc_auc["macro"]),
             color='orange',
             linestyle='--',
             lw=lw)

    plt.plot(ori_fpr["macro"], ori_tpr["macro"],
             label='PrCRS (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["macro"]),
             color='blue',
             linestyle='--',
             lw=lw)

#    plt.plot(lstm_fpr["macro"], lstm_tpr["macro"],
#             label='LSTM (AUC = {0:0.2f})'
#                   ''.format(lstm_roc_auc["macro"]),
#             color='black',
#             linestyle='--',
#             lw=lw)
    
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('AUROC',fontsize=20)
    plt.legend(loc="lower right",fontsize=20)
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\macro-average_ROC.png', dpi=500)
    
#def get_micro_acm_roc():
#    att_fpr, att_tpr, att_roc_auc = micro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\无迁移\\')
#    ori_fpr,ori_tpr,ori_roc_auc = micro(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\有迁移\\')
#
#    # Plot all ROC curves
#    lw=2
#    plt.figure()
#    plt.plot(att_fpr["micro"], att_tpr["micro"],
#             label='Meta-GNN (AUC = {0:0.2f})'
#                   ''.format(att_roc_auc["micro"]),
#             color='orange',
#             linestyle='--',
#             lw=lw)
#
#    plt.plot(ori_fpr["micro"], ori_tpr["micro"],
#             label='dglhan (AUC = {0:0.2f})'
#                   ''.format(ori_roc_auc["micro"]),
#             color='blue',
#             linestyle='--',
#             lw=lw)
#
#    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Cytokines')
#    plt.legend(loc="lower right")
#    # plt.show()
#    plt.savefig(r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code2\是否迁移效果比较\micro-average_ROC.png', dpi=500)

get_macro_acm_roc()
#get_micro_acm_roc()