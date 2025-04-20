import warnings
warnings.filterwarnings("ignore")
import numpy as np
import keras
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.data_utils import get_file

import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import csv
import time
import json
import pickle
import codecs
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from fl_server import obj_to_pickle_string, pickle_string_to_obj
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,roc_curve,auc
from dataset import gen_train_valid_data
import datetime,time
print("now is {}".format(datetime.datetime.today()))
datasource= gen_train_valid_data()

def getModel1():
    input_img = Input(shape=(118,))
    # # 分支0
    hidden = [64, 32, 16, 1, 16, 32, 64, 118]
    tower0 = input_img
    for nums in hidden:
        # tower0 = Dense(nums, activation=LeakyReLU(alpha=0.2) if nums != 1 else None)(tower0)
        tower0 = Dense(nums, activation=None if nums != 1 else None)(tower0)
        tower0 = LeakyReLU(alpha=0.2)(tower0)
        tower0 = Dropout(0.6)(tower0)
        if nums == 1:
            z0 = tower0
    # 把前面的计算逻辑，分别指定input和output，并构建成网络
    model = Model(inputs=input_img, outputs=tower0)
    # 编译model
    # adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss=keras.losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
    return model

# percent(剪枝率)
# 正常|规整剪枝标志

base_number =8# args.normal_regular
layers =9# args.layers
percent=0.5

if base_number <= 0:
    print('\r\n!base_number is error!\r\n')
    base_number = 1

model = getModel1()

print('旧模型: ', model)
total = 0
i = 0
for m in model.__module__():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            total += m.weight.data.shape[0]

# 确定剪枝的全局阈值
bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
y, j = torch.sort(bn)
thre_index = int(total * percent)
if thre_index == total:
    thre_index = total - 1
thre_0 = y[thre_index]

# ********************************预剪枝*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)

            if remain_channels == 0:
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))] = 1

            # ******************规整剪枝******************
            v = 0
            n = 1
            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)

                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_0.append(mask.shape[0])
            cfg.append(int(remain_channels))
            cfg_mask.append(mask.clone())
            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                  format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
pruned_ratio = float(pruned / total)
print('\r\n!预剪枝完成!')
print('total_pruned_ratio: ', pruned_ratio)


# ********************************预剪枝后model测试*********************************
def test():
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=64, shuffle=False, num_workers=1)
    model.eval()
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Accuracy: {:.2f}%\n'.format(acc))
    return


print('************预剪枝模型测试************')


# ********************************剪枝*********************************
newmodel =getModel1(cfg)

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
i = 0
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        if i < layers - 1:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w = m0.weight.data[:, idx0, :, :].clone()
            m1.weight.data = w[idx1, :, :, :].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
            m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
# ******************************剪枝后model测试*********************************
print('新模型: ', newmodel)
print('**********剪枝后新模型测试*********')
model = newmodel
test()
# ******************************剪枝后model保存*********************************
print('**********剪枝后新模型保存*********')
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, 'nin_prune.pth')
print('**********保存成功*********\r\n')

# *****************************剪枝前后model对比********************************
print('************旧模型结构************')
print(cfg_0)
print('************新模型结构************')
print(cfg, '\r\n')
