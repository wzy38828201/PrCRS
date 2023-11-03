import numpy as np
import torch
from torch.utils.data import DataLoader
from model import tranMedical
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix

torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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

# # 对数据集进行验证，返回准确率和loss
# def validate(model, data_loader, criterion, mode='train'):
#     model.eval()
#     losses = []
#     accs = []
#     with torch.no_grad():
#         for _, batch in enumerate(data_loader):
#             x, y = batch
#             x = x.cuda()
#             y = y.cuda()
#             y_pred = model(x)
#             loss = criterion(y_pred, y)
#             # print('loss: ', loss)
#             losses += [loss.item()]
#             # print(f'{mode}: y_pred before:  {y_pred}')
#             y_pred = F.softmax(y_pred, dim=-1)
#             y_pred = torch.argmax(y_pred, dim=-1).cpu().numpy()
#             # print(f'{mode}: y_pred after :  {y_pred}')
#             # print('y: ', y)
#             y = y.cpu().numpy()
#             accs += [list(y == y_pred)]
#
#     model.train()
#     average_loss = np.mean(np.array(losses))
#     average_acc = np.mean(np.array(accs))
#
#     return average_acc, average_loss

# 对数据集进行验证，返回准确率和loss
def validate(model, data_loader, criterion, thresh=0.5):  # 0.3
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
            x = x.cpu()#.cuda()
            y = y.cpu()#.cuda()
            y_pred = model(x)
            #y = y - 1
            loss = criterion(y_pred, y)
            losses += [loss.item()]

            y_pred[:, 0] = y_pred[:, 0] / 0.6156169
            y_pred[:, 1] = y_pred[:, 1] / 0.2515099
            y_pred[:, 2] = y_pred[:, 2] / 0.132873
            y_pred0 = F.softmax(y_pred, dim=-1)
            # y_pred0[:, 0] = y_pred0[:, 0] / 0.6156169
            # y_pred0[:, 1] = y_pred0[:, 1] / 0.2515099
            # y_pred0[:, 2] = y_pred0[:, 2] / 0.132873
            y_pred = torch.argmax(y_pred0, dim=-1).cpu().numpy()
            #y_pred = (y_pred0[:, 1].cpu().numpy() > thresh)
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
    average_mi = f1_score(y_labels, y_preds, average='micro')  # np.mean(np.array(micro_f1))
    average_pr = precision_score(y_labels, y_preds, average='weighted')  # np.mean(np.array(precision))
    average_re = recall_score(y_labels, y_preds, average='weighted')  # np.mean(np.array(recall))
    # tn, fp, fn, tp = confusion_matrix(y_labels, y_preds).ravel()
    # Specificity = tn / (tn + fp)

    return average_acc, average_loss, average_mi, average_pr, average_re#, Specificity

# def regularization_loss(model, factor=0.1, p=3):
#     '''
#     regularization_loss，只惩罚含有 weight 的参数
#     model: 传入模型
#     factor: 正则化惩罚系数
#     p: p-范数
#     '''
#     reg_loss = torch.tensor(0.,)
#     for name, w in model.named_parameters():
#         if 'weight' in name:    # 只对 参数名 含有 weight 的参数 正则化
#             reg_loss = reg_loss + torch.norm(w, p)
#     reg_loss = factor * reg_loss
#     return reg_loss

def train():
    # step 0: define outputdir以及tensorboard
    output_dir = './outputs1'
    sw = SummaryWriter(log_dir=output_dir)

    # step1: define classifier以及优化器以及损失函数
    model = tranMedical(num_class=3).cpu()#.cuda()
    #print('model: ', model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)#, eps=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # step2: def 训练dataloader，验证dataloader以及测试dataloader
    collate_fn = Collate()
    batch_size = 16#16
    trainset = Dataset('./data3/data1_train.npy')
    train_loader = DataLoader(trainset, num_workers=0,
                              shuffle=False, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    devset = Dataset('./data3/data1_valid.npy')
    dev_loader = DataLoader(devset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    testset = Dataset('./data3/data1_test.npy')
    test_loader = DataLoader(testset, num_workers=0,
                              shuffle=False, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    # step3: train and validate model
    steps = 0
    epoch = 50
    for i in range(epoch):
        for j, batch in enumerate(train_loader):
            optim.zero_grad()
            x, y = batch
            x = x.cpu()#.cuda()
            y = y.cpu()#.cuda()
            y_pred = model(x)
            print(y_pred,y)
            #y = y - 1
            loss = criterion(y_pred, y) # + regularization_loss(model)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optim.step()
            
            # 记录每一步的训练loss
            sw.add_scalar(f"train/loss_per_step", loss.item(), steps)
            steps += 1
            if j % 10 == 0:
                print(f'epoch: {i}  step: {j}  loss: {loss.item()}')

        # 测试训练集的平均loss以及准确率并将log写入tensorboard
        train_avg_acc, train_avg_loss, train_avg_mi, train_avg_pr, train_avg_re = validate(model, train_loader, criterion)
        sw.add_scalar(f"train/acc", train_avg_acc, i)
        sw.add_scalar(f"train/loss", train_avg_loss, i)
        sw.add_scalar(f"train/mi", train_avg_mi, i)
        sw.add_scalar(f"train/pr", train_avg_pr, i)
        sw.add_scalar(f"train/re", train_avg_re, i)
        print(f'train average loss: {train_avg_loss}  train average acc: {train_avg_acc}  train average micro: {train_avg_mi}  '
              f'train_avg_pr: {train_avg_pr}   train_avg_re: {train_avg_pr}')

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
        test_avg_acc, test_avg_loss, test_avg_mi, test_avg_pr, test_avg_re= validate(model, test_loader, criterion)
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

    # 从验证集确定最好的阈值
    best_mi = 0
    best_thresh = 0
    for t in range(0, 100, 5):
        t_ = t / 100.0
        dev_avg_acc, dev_avg_loss, dev_avg_mi, dev_avg_pr, dev_avg_re= validate(model, dev_loader, criterion, thresh=t_)
        if dev_avg_mi > best_mi:
            best_mi = dev_avg_mi
            best_thresh = t_

    print('\n\nBest Thresh: ', best_thresh)
    test_avg_acc, test_avg_loss, test_avg_mi, test_avg_pr, test_avg_re = validate(model, test_loader, criterion, thresh=best_thresh)

    print(f'Best test average loss: {test_avg_loss}  test average acc: {test_avg_acc}  test average micro: {test_avg_mi}  '
          f'test_avg_pr: {test_avg_pr}   test_avg_re: {test_avg_re}')
    # cm = confusion_matrix(all_labels, preds)
    # plot_confusion_matrix(cm, classes, savedir=savedir)
    # cm = confusion_matrix(all_labels, preds)
    # plot_confusion_matrix(cm, classes, savedir=savedir)

if __name__ == '__main__':
    train()