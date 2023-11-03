import numpy as np
import torch
from torch.utils.data import DataLoader
from model import tranMedical
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1234)

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


# 对数据集进行验证，返回准确率和loss
def validate(model, data_loader, criterion, thresh=0.3):#0.3
    model.eval()
    y_labels = []
    y_preds = []
    losses = []
    accs = []
    micro_f1 = []
    precision = []
    recall = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses += [loss.item()]

            y_pred0 = F.softmax(y_pred, dim=-1)
            # print('y_pred0: ', y_pred0)
            y_pred = torch.argmax(y_pred0, dim=-1).cpu().numpy()
            y_pred = (y_pred0[:,1].cpu().numpy() > thresh)
            y_pred = np.array([int(x) for x in y_pred])
            
            y = y.cpu().numpy()
            # print('y_pred: ', y_pred)
            # print('y: ', y)
            y_preds += list(y_pred)
            y_labels += list(y)

            # precision.append(precision_score(y, y_pred))
            # recall.append(recall_score(y, y_pred))
            # micro_f1.append(f1_score(y, y_pred, average='micro'))
            accs += [list(y == y_pred)]

    model.train()
    print('y_labels: ', y_labels)
    print('y_preds:  ', y_preds)
    average_loss = np.mean(np.array(losses))
    average_acc = np.mean(np.array(accs))
    average_mi = f1_score(y_labels, y_preds, average='micro') #np.mean(np.array(micro_f1))
    average_pr = precision_score(y_labels, y_preds) #np.mean(np.array(precision))
    average_re = recall_score(y_labels, y_preds) #np.mean(np.array(recall))

    return average_acc, average_loss, average_mi, average_pr, average_re

# 迁移学习的核心
# 固定住网络的某些参数，只对某些层训练    
def freeze_parameters(model, _type='all'):
    # [encoder,decoder,rnn]
    
    for k, v in model.named_parameters():
        v.requires_grad = True
    
    for k, v in model.fc.named_parameters():
        v.requires_grad = True
    
    for k, v in model.fftblock.named_parameters(): 
        v.requires_grad = True

    for k, v in model.fc1.named_parameters(): 
        v.requires_grad = True
    
# 加载分类层前面的参数,加载pt模型
def load_weight(model, ckpt_path):
    weight = torch.load(ckpt_path)['model']
    print('weight: ', weight.keys())
    for key in list(weight.keys()):
        if 'fc.' in key:
            del weight[key]
    model.load_state_dict(weight, strict=False)

def train():
    # step 0: define outputdir以及tensorboard
    output_dir = './output多'#s2
    sw = SummaryWriter(log_dir=output_dir)

    # step1: define classifier以及优化器以及损失函数
    model = tranMedical(num_class=2).cuda()
    # print('model: ', model)
    #ata1_ckpt = './outputs1/19.pt'
    #data1_ckpt = r'G:\图神经网络编程\深度学习预测CRS\迁移学习\code\outputs1\19.pt'
    # weight = torch.load(data1_ckpt)['model']
    # print('weight: ', weight.keys())
    # model.load_state_dict(torch.load(data1_ckpt)['model'], strict=False)

    # #这个是读取上个模型训练的数据模型内部的参数
    # load_weight(model, data1_ckpt)
    # freeze_parameters(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)#1e-4
    criterion = torch.nn.CrossEntropyLoss()

    # step2: def 训练dataloader，验证dataloader以及测试dataloader
    collate_fn = Collate()
    batch_size = 32#16
    trainset = Dataset('./data2/data2_train.npy')
    train_loader = DataLoader(trainset, num_workers=0,
                              shuffle=False, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    devset = Dataset('./data2/data2_valid.npy')
    dev_loader = DataLoader(devset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    testset = Dataset('./data2/data2_test.npy')
    test_loader = DataLoader(testset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    # step3: train and validate model
    steps = 0
    epoch = 50#20
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
                print(f'epoch: {i}  step: {j}  loss: {loss.item()}')

        #测试训练集的平均loss以及准确率并将log写入tensorboard
        train_avg_acc, train_avg_loss, train_avg_mi, train_avg_pr, train_avg_re = validate(model, train_loader, criterion)
        sw.add_scalar(f"train/acc", train_avg_acc, i)
        sw.add_scalar(f"train/loss", train_avg_loss, i)
        sw.add_scalar(f"train/mi", train_avg_mi, i)
        sw.add_scalar(f"train/pr", train_avg_pr, i)
        sw.add_scalar(f"train/re", train_avg_re, i)
        print(f'train average loss: {train_avg_loss}  train average acc: {train_avg_acc}  train average micro: {train_avg_mi}  '
              f'train_avg_pr: {train_avg_pr}   train_avg_re: {train_avg_re}')

        # 测试验证集的平均loss以及准确率并将log写入tensorboard
        dev_avg_acc, dev_avg_loss, dev_avg_mi, dev_avg_pr, dev_avg_re = validate(model, dev_loader, criterion)
        sw.add_scalar(f"dev/acc", dev_avg_acc, i)
        sw.add_scalar(f"dev/loss", dev_avg_loss, i)
        sw.add_scalar(f"dev/mi", dev_avg_mi, i)
        sw.add_scalar(f"dev/pr", dev_avg_pr, i)
        sw.add_scalar(f"dev/re", dev_avg_re, i)
        print(f'dev average loss: {dev_avg_loss}  dev average acc: {dev_avg_acc}  dev average micro: {dev_avg_mi}  '
              f'dev_avg_pr: {dev_avg_pr}   dev_avg_re: {dev_avg_re}')

        # 测试测试集集的平均loss以及准确率并将log写入tensorboard
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

    # # 从验证集确定最好的阈值
    # best_mi = 0
    # best_thresh = 0
    # for t in range(0, 100, 5):
    #     t_ = t / 100.0
    #     dev_avg_acc, dev_avg_loss, dev_avg_mi, dev_avg_pr, dev_avg_re = validate(model, dev_loader, criterion, thresh=t_)
    #     if dev_avg_mi > best_mi:
    #         best_mi = dev_avg_mi
    #         best_thresh = t_
    #
    # print('\n\nBest Thresh: ', best_thresh)
    # test_avg_acc, test_avg_loss, test_avg_mi, test_avg_pr, test_avg_re = validate(model, test_loader, criterion, thresh=best_thresh)
    #
    # print(f'Best test average loss: {test_avg_loss}  test average acc: {test_avg_acc}  test average micro: {test_avg_mi}  '
    #       f'test_avg_pr: {test_avg_pr}   test_avg_re: {test_avg_re}')
    ## cm = confusion_matrix(all_labels, preds)
    ## plot_confusion_matrix(cm, classes, savedir=savedir)

if __name__ == '__main__':
    train()