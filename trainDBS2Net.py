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
from LWENet import lwenet
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
    '--train-cover-dir', dest='train_cover_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/train/cover",
    # default=r"/data2/mingzhihu/dataset/MAE_suni4/train/cover",
)
parser.add_argument(
    '--valid-cover-dir', dest='valid_cover_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/val/cover",
    # default=r"/data2/mingzhihu/dataset/MAE_suni4/val/cover",
)
parser.add_argument(
    '--train-stego-dir', dest='train_stego_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/train/stego",
    # default=r"/data2/mingzhihu/dataset/BB_BOW_suni4/train/stego",
)
parser.add_argument(
    '--valid-stego-dir', dest='valid_stego_dir', type=str, required=False,
    default=r"/data2/mingzhihu/dataset/JPEG/BB-JUNI04-75/val/stego",
    # default=r"/data2/mingzhihu/dataset/BB_BOW_suni4/val/stego",
)
# parser.add_argument(
#     '--train-cover-dir', dest='train_cover_dir', type=str, required=False,
#     default=r"E:\zl\database\BossBase\JPEG\BB-JUNI04-75\train\cover",
# )
# # parser.add_argument(
# #     '--valid-cover-dir', dest='valid_cover_dir', type=str, required=False,
# #     default=r"E:\zl\database\BossBase\JPEG\BB-UERD04-95\test\cover",
# # )
# parser.add_argument(
#     '--train-stego-dir', dest='train_stego_dir', type=str, required=False,
#     default=r"E:\zl\database\BossBase\JPEG\BB-JUNI04-75/train/stego",
# )
# # parser.add_argument(
# #     '--valid-stego-dir', dest='valid_stego_dir', type=str, required=False,
# #     default=r"E:\zl\database\BossBase\JPEG\BB-JUNI04-95\test\stego",
# # )

# parser.add_argument(
#     '--valid-cover-dir', dest='valid_cover_dir', type=str, required=False,
#     default=r"E:\zl\database\ALASKA\ALASKA_jpeg95",
# )
# parser.add_argument(
#     '--valid-stego-dir', dest='valid_stego_dir', type=str, required=False,
#     default=r"E:\zl\database\ALASKA\JPEG\ALASKA-JUNI03-95",
# parser.add_argument('--finetune', dest='finetune', type=str, default="model_data/BOW_DBS2Net4_JMiPOD02_95/model_best.pth")
parser.add_argument('--finetune', dest='finetune', type=str, default=None)
parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, required=False,
                    default="model_data/DBS2Net_JUNI04_75")
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')  # 单卡16,这里单位是对，图片数应该乘以2
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=3407, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')

# cuda related
args = parser.parse_args()


# args.cuda = not args.no_cuda and torch.cuda.is_available()
def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
    log_file = osp.join(args.ckpt_dir, 'log.txt')
    log.configure_logging(file=log_file, root_handler_type=0)
    logger.info('Torch: {}'.format(str(torch.__version__)))
    logger.info('Command Line Arguments: {}'.format(str(args)))


setup(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

train_transform = transforms.Compose([utils.AugData(),
                                      utils.ToTensor()])  # 成对训练并使用图像增强（VA）时顺序必须为utils.AugData(),utils.ToTensor()，若只使用成对训练则utils.ToTensor()
val_transform = transforms.Compose([utils.ToTensor()])  # 成对训练

train_data = utils.DatasetPair(args.train_cover_dir, args.train_stego_dir, train_transform)
valid_data = utils.DatasetPair(args.valid_cover_dir, args.valid_stego_dir, val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = DBS2Net()
# logger.info('model: {}'.format(str(model)))
logger.info('Summary: {}'.format(str(summary(model.cuda(), (1, 256, 256)))))  # 打印模型参数
# print(model)
# print(summary(model.cuda(),(1,256,256)))
if args.cuda:
    model.cuda()
    # 多GPU时
    model = nn.DataParallel(model)
    device = torch.device('cuda:0')
    model.to(device)
cudnn.benchmark = True


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')


model.apply(initWeights)

params = model.parameters()
params_wd, params_rest = [], []
for param_item in params:
    if param_item.requires_grad:
        (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

param_groups = [{'params': params_wd, 'weight_decay': args.weight_decay},
                {'params': params_rest}]

optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(param_groups, lr=args.lr)
DECAY_EPOCH = [80, 150, 180]
# DECAY_EPOCH = [30, 60]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)

if args.finetune is not None:
    print("load!")
    model.load_state_dict(torch.load(args.finetune)['state_dict'], strict=True)


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


def train(epoch):
    total_loss = 0
    total_accu = 0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    logger.info('lr_train: {}'.format(str(lr_train)))
    # print(lr_train)

    model.train()
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data, label = data['images'].cuda(), data['labels'].cuda()
        data, label = Variable(data), Variable(label)

        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(os.listdir(args.train_cover_dir)) - args.batch_size * (len(train_loader) - 1)
            datas = data.view(last_batch_size * 2, 1, 256, 256)
            labels = label.view(last_batch_size * 2)
        else:
            datas = data.view(args.batch_size * 2, 1, 256, 256)
            labels = label.view(args.batch_size * 2)
        optimizer.zero_grad()
        output = model(datas)

        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss + loss.item()

        b_pred = output.max(1, keepdim=True)[1]
        b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()
        b_accu = b_correct / (labels.size(0))

        total_accu = total_accu + b_accu
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            # b_pred = output.max(1, keepdim=True)[1]
            # b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()
            #
            # b_accu=b_correct/(labels.size(0))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
            #     100. * (batch_idx+1) / len(train_loader),b_accu ,loss.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), b_accu, loss.item()))
    logger.info('train Epoch: {}\tavgLoss: {:.6f} \tavgAccu: {:.6f}'.format(epoch, total_loss / len(train_loader),
                                                                            total_accu / len(train_loader)))
    scheduler.step()


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
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
                datas = data.view(args.test_batch_size * 2, 1, 256, 256)
                labels = label.view(args.test_batch_size * 2)

            output = model(datas)

            output1 = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output1, labels)
            # print(loss)
            valid_loss = valid_loss + loss.item()

            # valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target,reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    valid_loss /= (len(valid_loader.dataset) * 2)
    logger.info('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(valid_loss, correct,
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
    logger.info('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        test_loss, correct, (len(valid_loader.dataset) * 2),
        100. * accu))
    S = len(valid_loader.dataset)  # 待测数据中所有的载密图像个数  具体的数量具体设置，如果载体图像等于载密数量则这样写代码即可
    C = len(valid_loader.dataset)  # 待测数据集中所有载体图像的个数
    FPR = (C - N) / C  # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
    Pmd = (S - P) / S  # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率
    logger.info('Valid set 虚警率(FPR): {}/{} ({:.6f}%)'.format(C - N, C, 100. * FPR))
    logger.info('Valid set 漏检率(FNR): {}/{} ({:.6f}%)'.format(S - P, S,
                                                             100. * Pmd))  # 名称定义来自于  来自于软件学报 论文 《基于深度学习的图像隐写分析综述》Journal of Software,2021,32(2):551−578 [doi: 10.13328/j.cnki.jos.006135]
    return accu, test_loss


t1 = time()
best_accuracy = 0.
for epoch in range(1, args.epochs+1):#
    # valid()
    train(epoch)
    accuracy,valid_loss = valid()
    # test()
    # torch.save(model.state_dict(), args.ckpt_dir+'/epoch_'+str(epoch)+'.pth',_use_new_zipfile_serialization = False)#保存每个epoc之后的网络参数
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        is_best = True
    else:
        is_best = False
    # if is_best:
    logger.info('Best accuracy: {}'.format(best_accuracy))
    save_checkpoint(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec': accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth'))


valid_mulit()

t2 = time()
print(t2 - t1)






