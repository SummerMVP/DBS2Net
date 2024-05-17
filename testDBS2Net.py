# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from model.DBS2Net import DBS2Net
import logging
import os.path as osp
from torch.backends import cudnn
import numpy as np
from time import *
import utils, log
import os
from torchsummary import summary

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--valid-cover-dir', dest='valid_cover_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/train/cover",
    # default=r"/data2/mingzhihu/dataset/JPEG/Gradient/coverE",

)

parser.add_argument(
    '--valid-stego-dir', dest='valid_stego_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/train/stego",
    # default=r"/data2/mingzhihu/dataset/JPEG/Gradient/stegoE",
)

parser.add_argument('--finetune', dest='finetune', type=str, default=None)

parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=3407, metavar='S',
                    help='random seed (default: 1)')

# cuda related
args = parser.parse_args()

def setup(args):
    args.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

setup(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

test_transform = transforms.Compose([utils.ToTensor()])  # 成对训练

test_data = utils.DatasetPair(args.valid_cover_dir, args.valid_stego_dir,test_transform)

valid_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = DBS2Net()
# model = SRM()
# print('model: {}'.format(str(model)))
# print('Summary: {}'.format(str(summary(model.cuda(), (1, 256, 256)))))  # 打印模型参数

if args.cuda:
    model.cuda()
    # 多GPU时
    model = nn.DataParallel(model)
    device = torch.device('cuda:0')
    model.to(device)
cudnn.benchmark = True

if args.finetune is not None:
    print("load!")
    model.load_state_dict(torch.load(args.finetune)['state_dict'], strict=True)


def CalGra():
    model.eval()
    valid_loss = 0
    correct = 0.
    # data.requires_grad = True
    for batch_idx, data in enumerate(valid_loader):
        if args.cuda:
            data, label = data['images'].cuda(), data['labels'].cuda()
        data, label = Variable(data), Variable(label)
        datas = data.view(args.test_batch_size * 2, 1, 256, 256)
        labels = label.view(args.test_batch_size * 2)
        datas.requires_grad = True
        output = model(datas)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)

    # print('loss:')
    # print(loss)
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output, label)

    model.zero_grad()
    loss.requires_grad_()
    loss.backward()

    # print("data.grad:")
    # print(data.grad)
    grad = datas.grad.data
    file_name = r'E:\dataset\Gradient\gradient.npy'

    np.save(file_name, grad.cpu())
    # print(grad)
    # print('grad.shape():')
    # print(grad.shape)


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            if args.cuda:
                data, label = data['images'].cuda(), data['labels'].cuda()
            data, label = Variable(data), Variable(label)

            if batch_idx == len(valid_loader) - 1:
                last_batch_size = len(os.listdir(args.valid_cover_dir)) - args.test_batch_size * (len(valid_loader) - 1)
                datas = data.view(last_batch_size * 2, 1, 256, 256)
                labels = label.view(last_batch_size * 2)
            else:
                datas = data.view(args.test_batch_size * 2, 1, 256, 256)
                labels = label.view(args.test_batch_size * 2)

            output = model(datas)

            output1 = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output1, labels)
            valid_loss = valid_loss + loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    valid_loss /= (len(valid_loader.dataset) * 2)
    print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(valid_loss, correct,
                                                                                    len(valid_loader.dataset) * 2,
                                                                                    100. * correct / (len(
                                                                                        valid_loader.dataset) * 2)))
    # print('valid set:  Accuracy: {}/{} ({:.6f}%)'.format(correct,len(valid_loader.dataset),100. * correct / len( valid_loader.dataset)))
    accu = float(correct) / (len(valid_loader.dataset) * 2)
    return accu, valid_loss


def sum(pred, target):
    # 此代码不包含统计所有的载体图片和所有载密图像的数量，需要在调用后设置所有载体图像的数量。
    # print(len(target))
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])
    # print(l1.count(0))
    # print(l1.count(2))
    # l1.count(0)即为 正确被判定为载体图像（阴性）的数量。l1.count(2)，即为正确被判定为载密图像（阳性）的数量。l1.count(0)+l1.count(2) 即为判断正确的总个数
    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2)


def valid_mulit():
    model.eval()
    test_loss = 0
    correct = 0.
    # accu = 0.
    N = 0  # 正确被分类为载体图像的数目
    P = 0  # 正确被分类为载密图像的数目
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            if args.cuda:
                data, label = data['images'].cuda(), data['labels'].cuda()
            # print("data.shape:")
            # print(data.shape)
            data, label = Variable(data), Variable(label)
            if batch_idx == len(valid_loader) - 1:
                last_batch_size = len(os.listdir(args.valid_cover_dir)) - args.test_batch_size * (len(valid_loader) - 1)
                datas = data.view(last_batch_size * 2, 1, 256, 256)
                labels = label.view(last_batch_size * 2)
            else:
                # print(data.shape)
                datas = data.view(args.test_batch_size * 2, 1, 256, 256)
                labels = label.view(args.test_batch_size * 2)
            # for data, target in test_loader:
            #     if args.cuda:
            #         data, target = data.cuda(), target.cuda()
            #     data, target = Variable(data), Variable(target)
            output = model(datas)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, labels, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            a, b, c = sum(pred, labels)
            N += a  # 正确判断成cover
            P += b  # 正确判断成stego
            correct += c  # 判断正确的数量
    test_loss /= len(valid_loader.dataset)
    accu = float(correct) / (len(valid_loader.dataset) * 2)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        test_loss, correct, (len(valid_loader.dataset) * 2),
        100. * accu))
    S = len(valid_loader.dataset)  # 待测数据中所有的载密图像个数  具体的数量具体设置，如果载体图像等于载密数量则这样写代码即可
    C = len(valid_loader.dataset)  # 待测数据集中所有载体图像的个数
    FPR = (C - N) / C  # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
    Pmd = (S - P) / S  # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率
    print('Valid set 虚警率(FPR): {}/{} ({:.6f}%)'.format(C - N, C, 100. * FPR))
    print('Valid set 漏检率(FNR): {}/{} ({:.6f}%)'.format(S - P, S,
                                                             100. * Pmd))  # 名称定义来自于  来自于软件学报 论文 《基于深度学习的图像隐写分析综述》Journal of Software,2021,32(2):551−578 [doi: 10.13328/j.cnki.jos.006135]
    return accu, test_loss


t1 = time()

valid_mulit()
# CalGra()

t2 = time()
print(t2 - t1)






