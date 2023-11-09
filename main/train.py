import numpy as np
import torch
from torch.utils.data import DataLoader
from model import tranMedical
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt


# 定义dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.dataset = np.load(dataset_file, allow_pickle=True)

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

# 对数据按长度进行padding
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


def validate(model, data_loader, criterion):
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses += [loss.item()]

            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = torch.argmax(y_pred, dim=-1).cpu().numpy()
            # print('y_pred: ', y_pred)
            y = y.cpu().numpy()
            accs += [list(y == y_pred)]

    model.train()
    average_loss = np.mean(np.array(losses))
    average_acc = np.mean(np.array(accs))
    
    return average_acc, average_loss
    

def train():
    # step 0: define outputdir以及tensorboard
    output_dir = './outputs'
    sw = SummaryWriter(log_dir=output_dir)

    # step1: define classifier以及优化器以及损失函数
    model = tranMedical(num_class=4).cuda()
    print('model: ', model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # step2: def 训练dataloader，验证dataloader以及测试dataloader
    collate_fn = Collate()
    batch_size = 16
    trainset = Dataset('./data/data2_train.npy')
    train_loader = DataLoader(trainset, num_workers=2,
                              shuffle=True, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    devset = Dataset('./data/data2_valid.npy')
    dev_loader = DataLoader(devset, num_workers=2,
                              shuffle=True, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    testset = Dataset('./data/data2_test.npy')
    test_loader = DataLoader(testset, num_workers=2,
                              shuffle=True, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    # step3: train and validate model
    steps = 0
    epoch = 20
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

        # 测试训练集的平均loss以及准确率并将log写入tensorboard
        train_avg_acc, train_avg_loss = validate(model, train_loader, criterion)
        sw.add_scalar(f"train/acc", train_avg_acc, i)
        sw.add_scalar(f"train/loss", train_avg_loss, i)
        print(f'train average loss: {train_avg_loss}  train average acc: {train_avg_acc}')

        # 测试验证集的平均loss以及准确率并将log写入tensorboard
        dev_avg_acc, dev_avg_loss = validate(model, dev_loader, criterion)
        sw.add_scalar(f"dev/acc", dev_avg_acc, i)
        sw.add_scalar(f"dev/loss", dev_avg_loss, i)
        print(f'dev average loss: {dev_avg_loss}  dev average acc: {dev_avg_acc}')


        # 测试测试集集的平均loss以及准确率并将log写入tensorboard
        test_avg_acc, test_avg_loss = validate(model, test_loader, criterion)
        sw.add_scalar(f"test/acc", test_avg_acc, i)
        sw.add_scalar(f"test/loss", test_avg_loss, i)
        print(f'test average loss: {test_avg_loss}  test average acc: {test_avg_acc}')

        checkpoint = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": int(i),
        }
        torch.save(checkpoint, os.path.join(output_dir, f'{i}.pt'))

    # cm = confusion_matrix(all_labels, preds)
    # plot_confusion_matrix(cm, classes, savedir=savedir)

if __name__ == '__main__':
    train()