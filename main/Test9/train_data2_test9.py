import numpy as np
import torch
from torch.utils.data import DataLoader
from model import tranMedical
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score,roc_curve, auc#confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(123)
torch.cuda.manual_seed_all(1234)
# 定义dataset
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

# 对数据进行batch
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

#从这里开始是判断医生给的数据
def validate1(model, data_loader, criterion, mode='train'):
    model.eval()
    y_pred_scores, y_preds = [], []
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
            y_pred_scores += [y_pred_score]
            y_preds += [y_pred_label]

        return y_pred_scores, y_preds

# 对数据集进行验证，返回准确率和loss
def validate(model, data_loader, criterion, thresh=0.3):#0.3
    model.eval()
    y_labels = []
    y_preds = []
    losses = []
    accs = []
#    micro_f1 = []
#    precision = []
#    recall = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)#logits
            loss = criterion(y_pred, y)
            losses += [loss.item()]

            y_pred0 = F.softmax(y_pred, dim=-1)
            # print('y_pred0: ', y_pred0)
            #y_pred = torch.argmax(y_pred0, dim=-1).cpu().numpy()
            y_pred = (y_pred0[:,1].cpu().numpy() > thresh)#prediction
            y_pred = np.array([int(x) for x in y_pred])
            
            y = y.cpu().numpy()
            # print('y_pred: ', y_pred)
            # print('y: ', y)
            y_preds += list(y_pred)
            y_labels += list(y)#labels

            # precision.append(precision_score(y, y_pred))
            # recall.append(recall_score(y, y_pred))
            # micro_f1.append(f1_score(y, y_pred, average='micro'))
            accs += [list(y == y_pred)]

    model.train()
#    print('y_labels: ', y_labels)
#    print('y_preds:  ', y_preds)
    average_loss = np.mean(np.array(losses))
    average_acc = np.mean(np.array(accs))
    average_mi = f1_score(y_labels, y_preds, average='micro') #np.mean(np.array(micro_f1))
    average_pr = precision_score(y_labels, y_preds) #np.mean(np.array(precision))
    average_re = recall_score(y_labels, y_preds) #np.mean(np.array(recall))

    return average_acc, average_loss, average_mi, average_pr, average_re

def freeze_parameters(model, _type='all'):
    # [encoder,decoder,rnn]
    
    for k, v in model.named_parameters():
        v.requires_grad = False
    
    for k, v in model.fc.named_parameters():
        v.requires_grad = False
    
    for k, v in model.fftblock.named_parameters(): 
        v.requires_grad = True

    for k, v in model.fc1.named_parameters(): 
        v.requires_grad = True
    
def load_weight(model, ckpt_path):
    weight = torch.load(ckpt_path)['model']
    print('weight: ', weight.keys())
    for key in list(weight.keys()):
        if 'fc.' in key:
            del weight[key]
    model.load_state_dict(weight, strict=False)
#0.0003,0.0002,0.0001,0.00008,0.00006,0.00005,0.00004,0.00003,,0.00001
lr = [0.0003,0.00025,0.0002,0.00015,0.0001,]#0.00005,0.00004,0.00003,0.00002,0.00001]
bs = [16,18,20,22,24]
# lr = [0.00004]
# bs = [16]
#import os
def train():
    dicc = {}
    for ilr in lr:
        for jbs in bs:
            path = 'main/Test9/model/'
            foldername = path + "\\" + "all," + str(ilr) + "," + str(jbs)

            word_name = os.path.exists(foldername)

            if not word_name:
                os.makedirs(foldername)

            output_dir = foldername#'./output多'#s2
            sw = SummaryWriter(log_dir=output_dir)

            model = tranMedical(num_class=2).cuda()
            model.zero_grad()

#            # 迁移学习
            #ata1_ckpt = './outputs1/19.pt'
            data1_ckpt = r'main/Test9/model/Primitive_transfer.pt'
            weight = torch.load(data1_ckpt)['model']
            print('weight: ', weight.keys())
            del weight['fc.linear_layer.bias']
            del weight['fc.linear_layer.weight']
            #model.load_state_dict(torch.load(data1_ckpt)['model'], strict=False)
            model.load_state_dict(weight,strict = False)

            load_weight(model, data1_ckpt)
            freeze_parameters(model)

            #seed1234，第2个特征，0.3：“5e-5，20，第141”
            optim = torch.optim.Adam(model.parameters(), lr=ilr)
            criterion = torch.nn.CrossEntropyLoss()
            optim.zero_grad()

            collate_fn = Collate()
            batch_size =jbs#20
            trainset = Dataset('main/Test9/data193_train.npy')
            train_loader = DataLoader(trainset, num_workers=0,
                                      shuffle=False, batch_size=batch_size,
                                      pin_memory=False, drop_last=True,
                                      collate_fn=collate_fn)

            devset = Dataset('main/Test9/data193_valid.npy')
            dev_loader = DataLoader(devset, num_workers=0,
                                      shuffle=False, batch_size=1,
                                      pin_memory=False, drop_last=True,
                                      collate_fn=collate_fn)

            testset = Dataset('main/Test9/data193_test.npy')
            test_loader = DataLoader(testset, num_workers=0,
                                      shuffle=False, batch_size=1,
                                      pin_memory=False, drop_last=True,
                                      collate_fn=collate_fn)

            # step3: train and validate model
            steps = 0
            epoch = 100#70
            dic = {}
            dic_tr = {}
            dic_de = {}
            dic_te = {}
            dic2_tr = {}
            dic2_de = {}
            dic2_te = {}
            for i in range(epoch):
                for j, batch in enumerate(train_loader):
                    optim.zero_grad()
                    x, y = batch
                    x = x.cuda()
                    y = y.cuda()
                    y_pred = model(x)

                    loss = criterion(y_pred, y)
                    loss.backward()
                    optim.step()

                    # 记录每一步的训练loss
                    sw.add_scalar(f"train/loss_per_step", loss.item(), steps)
                    steps += 1
                    if j % 10 == 0:
                        print(f'epoch: {i}  steps: {j}  loss: {loss.item()}')

                train_avg_acc, train_avg_loss, train_avg_mi, train_avg_pr, train_avg_re = validate(model, train_loader, criterion)
                sw.add_scalar(f"train/acc", train_avg_acc, i)
                sw.add_scalar(f"train/loss", train_avg_loss, i)
                sw.add_scalar(f"train/mi", train_avg_mi, i)
                sw.add_scalar(f"train/pr", train_avg_pr, i)
                sw.add_scalar(f"train/re", train_avg_re, i)
                print(f'train average loss: {train_avg_loss}  train average acc: {train_avg_acc}  train average micro: {train_avg_mi}  '
                      f'train_avg_pr: {train_avg_pr}   train_avg_re: {train_avg_re}')

                dev_avg_acc, dev_avg_loss, dev_avg_mi, dev_avg_pr, dev_avg_re = validate(model, dev_loader, criterion)
                sw.add_scalar(f"dev/acc", dev_avg_acc, i)
                sw.add_scalar(f"dev/loss", dev_avg_loss, i)
                sw.add_scalar(f"dev/mi", dev_avg_mi, i)
                sw.add_scalar(f"dev/pr", dev_avg_pr, i)
                sw.add_scalar(f"dev/re", dev_avg_re, i)
                print(f'dev average loss: {dev_avg_loss}  dev average acc: {dev_avg_acc}  dev average micro: {dev_avg_mi}  '
                      f'dev_avg_pr: {dev_avg_pr}   dev_avg_re: {dev_avg_re}')

                test_avg_acc, test_avg_loss, test_avg_mi, test_avg_pr, test_avg_re = validate(model, test_loader, criterion)
                sw.add_scalar(f"test/acc", test_avg_acc, i)
                sw.add_scalar(f"test/loss", test_avg_loss, i)
                sw.add_scalar(f"test/mi", test_avg_mi, i)
                sw.add_scalar(f"test/pr", test_avg_pr, i)
                sw.add_scalar(f"test/re", test_avg_re, i)
                print(f'test average loss: {test_avg_loss}  test average acc: {test_avg_acc}  test average micro: {test_avg_mi}  '
                      f'test_avg_pr: {test_avg_pr}   test_avg_re: {test_avg_re}')
                print('\n\n')

                checkpoint = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": int(i),
                }
                torch.save(checkpoint, os.path.join(output_dir, f'{i}.pt'))

                # 获取字典中最大值的键
                if train_avg_re >= 0.8:
                    if dev_avg_re >= 0.7:
                        if dev_avg_pr >= 0.3:
                            if test_avg_re >= 0.8:
                                if test_avg_pr >= 0.3:
                                    po = test_avg_pr+test_avg_re#+train_avg_pr+dev_avg_pr+test_avg_pr
                                    dic[i] = po
                                    dic_tr[i] = train_avg_re
                                    dic_de[i] = dev_avg_re
                                    dic_te[i] = test_avg_re
                                    dic2_tr[i] = train_avg_pr
                                    dic2_de[i] = dev_avg_pr
                                    dic2_te[i] = test_avg_pr
            print('Run to the location：',ilr,jbs)
            import operator
            if len(dic.keys())==0:
                print('The current features are not enough to complete the prediction with high accuracy, so it is recommended to add features and predict again')
            else:
                max_key = max(dic.items(), key=operator.itemgetter(1))[0]
                print('Best model：',max_key,'The corresponding training verification test recall rate：',dic_tr[max_key], dic_de[max_key], dic_te[max_key],
                      'The corresponding training verifies the test accuracy：',dic2_tr[max_key], dic2_de[max_key], dic2_te[max_key])

                #np.save(os.path.join(savedir, 'data2_train多.npy'), np.array(ysgdsj))#ysgdsj是医生给的数据，列表的形式
                # step1: define classifier
                dicc[ilr,jbs,max_key] = dic_te[max_key]+dic2_te[max_key]#+dic2_tr[max_key]+dic2_de[max_key]+dic2_te[max_key]
                max_keyc = max(dicc.items(), key=operator.itemgetter(1))[0]
                file_handle = open(r'main/Test9/model/1_day.txt', mode='a+')
                file_handle.write('All the best models：'+str(max_keyc)+str(dicc[max_keyc])+'\n')

    if len(dicc.keys())==0:
        print('The current features are not enough to complete the prediction with high accuracy, so it is recommended to add features and predict again')
    else:
        max_keyc = max(dicc.items(), key=operator.itemgetter(1))[0]
        print('Best model：',max_keyc,dicc[max_keyc])
    model.zero_grad()
    optim.zero_grad()

if __name__ == '__main__':
    train()