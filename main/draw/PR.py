import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score,recall_score
from matplotlib import rcParams
import pandas as pd
import numpy as np
from sklearn import metrics

a = "main/Model comparison/Squeezeformer//" 
b = "main/Model comparison/PrCRS//" 
c = "main/Model comparison/CNN//"  
e = "main/Model comparison/Transformer//"  


def macro(path):
    y_test = pd.read_csv(path + 'labels.csv', header=None)
    y_test = y_test[0].tolist()
    
    y_score = pd.read_csv(path + 'prediction.csv', header=None)
    y_score = y_score[0].tolist()

    precisionx = precision_score(y_test, y_score, average='macro')#, average='weighted'
    recallx = recall_score(y_test, y_score, average='macro')

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
    precision[1] = precisionx
    recall[1] = recallx

    roc_prc = metrics.auc(recall, precision)
    #print(precision,recall,roc_prc)
    return precision, recall,roc_prc

config = {
    "font.family":'Times New Roman',
    "font.size":30
#     "mathtext.fontset":'stix',
}
rcParams.update(config)
 
precision1, recall1, roc_prc1 = macro(a)
precision2, recall2, roc_prc2 = macro(b)
precision3, recall3, roc_prc3 = macro(c)
#precision4, recall4, roc_prc4 = macro(d)
precision5, recall5, roc_prc5 = macro(e)


#plt.plot(recall1, precision1, linewidth=4, color="red")
#plt.xlabel("Recall", fontsize=12, fontweight='bold')
#plt.ylabel("Precision", fontsize=12, fontweight='bold')
#plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")

plt.figure(figsize=(10,8))
lw = 2

plt.plot(recall3,precision3, 
         color='orange',
         lw=lw, 
         label=' CNN = %0.2f' % roc_prc3,
         linestyle='--') 

plt.plot(recall5,precision5, 
         color='purple',
         lw=lw, 
         label=' Transformer = %0.2f' % roc_prc5,
         linestyle='--')

plt.plot(recall1,precision1, 
         color='blue',
         lw=lw, 
         label=' Squeezeformer = %0.2f' % roc_prc1,
         linestyle='--') 

plt.plot(recall2,precision2, 
         color='red',
         lw=lw, 
         label=' PrCRS = %0.2f' % roc_prc2,
         linestyle='--') 

#plt.plot(recall4,precision4, 
#         color='black',
#         lw=lw, 
#         label=' LSTM = %0.4f' % roc_prc4,
#         linestyle='--') 



plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30) 
plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title(' AUPRC',fontsize=20)
plt.legend(loc="lower right",fontsize=20)


bwith = 1.0 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig(r'main/Model comparison/PRC.png',dpi=600)
plt.show()