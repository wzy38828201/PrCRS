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
        # if np.isin(np.nan, feat):
        #     print('label: ', label)
        #     print('feats: ', np.isin(np.nan, feat), feat)
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


def validate_check(model, data_loader, criterion, mode='train'):
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
    
# 固定住网络的某些参数，只对某些层训练    
def freeze_parameters(model, _type='all'):
    # [encoder,decoder,rnn]
    
    for k, v in model.named_parameters():
        v.requires_grad = False
    
    for k, v in model.fc.named_parameters():
        v.requires_grad = True
    
    for k, v in model.fftblock.named_parameters(): 
        v.requires_grad = True

    for k, v in model.fc1.named_parameters(): 
        v.requires_grad = True
    
# 加载分类层前面的参数    
def load_weight(model, ckpt_path):
    weight = torch.load(ckpt_path)['model']
    print('weight: ', weight.keys())
    for key in list(weight.keys()):
        if 'fc.' in key:
            del weight[key]
    model.load_state_dict(weight, strict=False)

def train():
    # step 0: define outputdir以及tensorboard
    output_dir = './outputs2'
    sw = SummaryWriter(log_dir=output_dir)

    # step1: define classifier以及优化器以及损失函数
    model = tranMedical(num_class=2).cuda()
    print('model: ', model)
    data1_ckpt = './outputs1/19.pt'
    # weight = torch.load(data1_ckpt)['model']
    # print('weight: ', weight.keys())
    # model.load_state_dict(torch.load(data1_ckpt)['model'], strict=False)
    load_weight(model, data1_ckpt)
    freeze_parameters(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # step2: def 训练dataloader，验证dataloader以及测试dataloader
    collate_fn = Collate()
    batch_size = 1
    trainset = Dataset('./data3/data2_train.npy')
    train_loader = DataLoader(trainset, num_workers=2,
                              shuffle=True, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    devset = Dataset('./data3/data2_valid.npy')
    dev_loader = DataLoader(devset, num_workers=2,
                              shuffle=True, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    testset = Dataset('./data3/data2_test.npy')
    test_loader = DataLoader(testset, num_workers=2,
                              shuffle=True, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    # step3: train and validate model
    steps = 0
    epoch = 1
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

        if i == 0:
            y_pred_scores, y_preds, y_labels = validate_check(model, train_loader, criterion)
            print(f'y_pred_scores: {y_pred_scores}')
            print(f'y_preds:       {y_preds}')
            print(f'y_labels:      {y_labels}')
            print(f'positive: ', np.sum(np.array(y_labels)))

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

        y_pred_scores, y_preds, y_labels = validate_check(model, test_loader, criterion)
        print(f'y_pred_scores: {y_pred_scores}')
        print(f'y_preds:       {y_preds}')
        print(f'y_labels:      {y_labels}')

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