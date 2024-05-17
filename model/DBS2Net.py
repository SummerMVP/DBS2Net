import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from math import floor, ceil

SRM_npy = np.load(r'SRM_Kernels.npy')

class L2_nrom(nn.Module):
    def __init__(self,mode='l2'):
        super(L2_nrom, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True)).pow(0.5)
            norm = embedding / (embedding.pow(2).mean(dim=1, keepdim=True)).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x)
            embedding = _x.sum((2,3), keepdim=True)
            norm = embedding / (torch.abs(embedding).mean(dim=1, keepdim=True))
        return norm

class Sepconv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Sepconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)


    def forward(self, input):
        out1 = self.conv1(input)
        out = self.conv2(out1)
        return out


class _Transition1(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition1, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))


# Sepconv+Conv+无池化[标准]
class DenseNet_Add_3(nn.Module):
    def __init__(self, num_layers=6):
        super(DenseNet_Add_3,self).__init__()

        # 高通滤波 卷积核权重初始化
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=True)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy
        # self.conv = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2)

        self.features = nn.Sequential(OrderedDict([('norm0', nn.BatchNorm2d(30)), ]))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        self.DC_branchl = DC_branchl(2, in_ch=30, num_module=num_layers)

        self.trans1 = Sepconv(30,30)   # BlockB
        self.trans2 = _Transition1(num_input_features=30,num_output_features=2)

    def forward(self, input):
        HPF_output = F.conv2d(input, self.srm_filters_weight, stride=1, padding=2)
        # HPF_output = self.conv(input)
        # np.save('F:\denoise\code\LWENet\model_data/feature/HPF_output-95.npy', HPF_output.cpu().detach().numpy())
        output = self.features(HPF_output)
        output = self.DC_branchl(output)
        # np.save('F:\denoise\code\LWENet\model_data\LWENet2_JUNI04_95/DBS.npy', output.cpu().detach().numpy())
        output1 = self.trans1(output)
        output2 = self.trans2(output)
        # print(output1.shape) #[2, 30, 256, 256]
        # print(output2.shape) #[2, 2, 128, 128]
        output = torch.cat([output1, output2], dim=1)
        # np.save('model_data/feature/DBS2Net-75-100.npy', output.cpu().detach().numpy())

        return output



class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.BatchNorm2d(in_ch)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.BatchNorm2d(in_ch)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.BatchNorm2d(in_ch)]
        ly += [nn.ReLU(inplace=True)]


        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.BatchNorm2d(in_ch)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):  # 空洞卷积层
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride,
                         dilation=stride)]  # dilation 参数表示卷积操作中的空洞（dilated）卷积的扩张率。空洞卷积是一种通过在卷积核的元素之间引入间隔来增加感受野大小的技术。
        ly += [nn.BatchNorm2d(in_ch)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.BatchNorm2d(in_ch)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)  # 残差连接


class CentralMaskedConv2d(nn.Conv2d):  # 盲点网络
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0  # 最中间的像素不可见

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class SPP2d(nn.Module):
    def __init__(self, num_level, pool_type='avg_pool'):
        super(SPP2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):
        res = []
        N, C, H, W = x.shape
        for i in range(self.num_level):
            level = i+1
            # if i == 0:
            #     level = i +1
            # else:
            #     level = 2 * i
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)

            if i == 0:
                res = tensor.view(N, -1)
            else:
                # res = tensor
                res = torch.cat((res, tensor.view(N, -1)), 1)
        return res

class DBSNet_Add(nn.Module):
    def __init__(self, base_ch=30,num_module=6):
        super(DBSNet_Add, self).__init__()

        # 高通滤波 卷积核权重初始化
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=False)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy
        self.relu = nn.ReLU(inplace=True)
        # 开始加DBPNet
        #preprocessing
        self.branch1 = DC_branchl(2, base_ch, num_module)

        ly = []
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, 1, kernel_size=1)]
        self.tail = nn.Sequential(*ly)


    def forward(self, input):
        HPF_output = self.relu(F.conv2d(input, self.srm_filters_weight, stride=1, padding=2))
        output = self.branch1(HPF_output)
        output = self.tail(output)
        # np.save('model_data/DBPS_loss_LWENet_juni4_75/tail.npy', output.cpu().detach().numpy())
        return output

# 标准
class DBS2Net(nn.Module):
    def __init__(self):
        super(DBS2Net4, self).__init__()
        #preprocessing+BlockB
        self.Dense_layers  = DenseNet_Add_3(num_layers=6)


        #feature extraction
        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, padding=1) #BlockC
        self.layer5_BN = nn.BatchNorm2d(32)
        self.layer5_AC = nn.ReLU()

        self.layer6 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#BlockC
        self.layer6_BN = nn.BatchNorm2d(64)
        self.layer6_AC = nn.ReLU()

        self.avgpooling2 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#BlockC
        self.layer7_BN = nn.BatchNorm2d(64)
        self.layer7_AC = nn.ReLU()

        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#BlockC
        self.layer8_BN  = nn.BatchNorm2d(128)
        self.layer8_AC = nn.ReLU()

        self.avgpooling3 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#BlockC

        self.layer9_BN = nn.BatchNorm2d(128)
        self.layer9_AC = nn.ReLU()

        # self.layer10 = nn.Conv2d(128,256,kernel_size=3, padding=1)#BlockD,如果是正常卷积，难以收敛？
        self.layer10 = Sepconv(128, 256)  # BlockD
        self.layer10_BN = nn.BatchNorm2d(256)
        self.layer10_AC = nn.ReLU()
        #MGP
        self.GAP1 = nn.AdaptiveAvgPool2d((1,1))
        # self.GAP2 = nn.AdaptiveMaxPool2d((1,1))#反而降低准确率
        self.L2_norm = L2_nrom(mode='l2')
        # self.L1_norm = L2_nrom(mode='l1')
        # #classifier
        # self.fc1 = nn.Linear(256*3, 2)
        self.fc1 = nn.Linear(256*2, 2)


    def forward(self, input):

        Dense_block_out = self.Dense_layers(input)
        layer5_out = self.layer5(Dense_block_out)
        layer5_out = self.layer5_BN(layer5_out)
        layer5_out = self.layer5_AC(layer5_out)

        layer6_out = self.layer6(layer5_out)
        layer6_out = self.layer6_BN(layer6_out)
        layer6_out = self.layer6_AC(layer6_out)

        avg_pooling2 = self.avgpooling2(layer6_out)

        layer7_out = self.layer7(avg_pooling2)
        layer7_out = self.layer7_BN(layer7_out)
        layer7_out = self.layer7_AC(layer7_out)

        layer8_out = self.layer8(layer7_out)
        layer8_out = self.layer8_BN(layer8_out)
        layer8_out = self.layer8_AC(layer8_out)

        avg_pooling3 = self.avgpooling2(layer8_out)

        layer9_out = self.layer9(avg_pooling3)
        layer9_out = self.layer9_BN(layer9_out)
        layer9_out = self.layer9_AC(layer9_out)

        layer10_out = self.layer10(layer9_out)
        layer10_out = self.layer10_BN(layer10_out)
        layer10_out = self.layer10_AC(layer10_out)

        output_GAP1 = self.GAP1(layer10_out)
        # output_GAP2 = self.GAP2(layer10_out)
        output_GAP1 = output_GAP1.view( -1,256)
        # output_GAP2 = output_GAP2.view(-1, 256)
        output_L2 = self.L2_norm(layer10_out)
        # output_L1 = self.L1_norm(layer10_out)
        # output_GAP = output_GAP.view( -1,256)
        output_L2 = output_L2.view( -1,256)
        # output_L1 = output_L1.view(-1, 256)
        Final_feat = torch.cat([output_GAP1,output_L2],dim=-1)
        output = self.fc1(Final_feat)
        return output


# if __name__ == '__main__':
#     from torchsummary import summary
#     Input = torch.randn(1, 1, 256, 256).cuda()
#     net = DBS2Net().cuda()
#     print(summary(net,(1,256,256)))