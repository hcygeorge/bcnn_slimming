#%%
import os
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchsummary import summary

from load_birds import train_data_load, test_data_load
from load_cars import LoadCars
from load_aircraft import LoadAircrafts
from models import VGG, BCNN, BCNNMultiModel, ImproveBCNN, BCNNResnet
from ptflops import get_model_complexity_info
from utils import savemodel

import torch.nn.functional as F
#%%
# parser = argparse.ArgumentParser(description='PyTorch Slimming main')
# parser.add_argument('--pruned', default='', type=str, metavar='PATH',
#                     help='path to pruned model (default: none)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to checkpoint or pruned model (default: none)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#%%
# dataset
dataset = 'Aircrafts'
BATCH = 48
WORKERS = 0
# model
pruned = ''  #args.pruned
resume = 'D:/model/bestmodel0504_trial07a.pkl' #args.resume
data_class = {'Aircrafts': 100, 'Birds': 200, 'Cars': 196}
num_classes = data_class[dataset]
cuda = True

percent = 40


# bestmodel0504_trial07a
# bestmodel0507_trial07b
# bestmodel0525_trial07c
#%%
# Model
print("Loading model...")
if pruned:
    pruned_pkl = torch.load(pruned, map_location='cpu')
    model = BCNN(num_classes, cfg=pruned_pkl['cfg'])
    # bone = torch.load(bone, map_location='cpu')
    # model = BCNN(num_classes, cfg=bone['cfg'])
    state_dict = OrderedDict([('classifier.weight', v) if k == 'fc.weight' else (k, v) for k, v in pruned_pkl['state_dict'].items()])  # or pruned_pkl['state_dict']
    state_dict = OrderedDict([('classifier.bias', v) if k == 'fc.bias' else (k, v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
else:
    checkpoint = torch.load(resume, map_location='cpu')
    model = BCNN(num_classes)
    state_dict = OrderedDict([('classifier.weight', v) if k == 'fc.weight' else (k, v) for k, v in checkpoint['state_dict'].items()])
    state_dict = OrderedDict([('classifier.bias', v) if k == 'fc.bias' else (k, v) for k, v in state_dict.items()])
    print("Load the checkpoint")
    start_epoch = checkpoint['epoch'] + 1
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(state_dict)
    print("Path:'{}'\nLast epoch: {} \nValidAcc: {:.2f}%"
        .format(resume, checkpoint['epoch'], best_prec1))

if cuda:
    model.cuda()

#%%
# Dataloader
if dataset == 'Birds':
    data_path = 'C:/Dataset/Birds/'
    ROOT_TEST = data_path + 'test/'
    label_test, test_loader = test_data_load(ROOT_TEST, BATCH, WORKERS)
elif dataset == 'Cars':
    cars = LoadCars()
    test_loader = cars.load_data(False, BATCH, WORKERS)
elif dataset == 'Aircrafts':
    aircrafts = LoadAircrafts()
    test_loader = aircrafts.createloader(dataset='test', batch=BATCH, workers=WORKERS)

#%%
# Test model

def testaccu(model, loader):
    with torch.no_grad():
        kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
        correct = 0
        if __name__ == '__main__':
            model.eval()
            for data, target in test_loader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(data)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            print('\nTest accuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            # return correct / float(len(test_loader.dataset))
            
def testaccu2(model, loader, cuda=False):
    valid_correct = 0
    valid_total = 0 
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            
            out = model(data)  # forward
            # valid_loss = criterion(out, target)
            _, pred = torch.max(out.data, 1)  # max() return maximum and its index in each row
            valid_total += float(target.size(0))
            valid_correct += float((pred == target).sum())
            
    valid_acc = 100*valid_correct / valid_total
    print(valid_acc)

#%%
# model = BCNN(num_classes=200,
#              cfg=[23, 58, 'M', 89, 118, 'M', 179, 214, 250, 'M', 434, 443, 457, 'M', 372, 281, 461, 'M'])
# if cuda:
#     model.cuda()
#%%
# if __name__ == '__main__':
# Accu
print('Dataset: {}\nModel: {}'.format(
    dataset, pruned.split('/')[-1]))
print('Testing model...')
accu = testaccu2(model, test_loader, cuda)
# print('Done.\n')

#%%
# Parameters
summary(model, input_size=(3, 224, 224), batch_size=48, device='cuda')

#%%
# GMacs 
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#%%
# CFG
print(pruned_pkl['cfg'])
#%%
# Test
    # for idx, m in model.named_parameters():
    #     print(idx, m.size())
    
    # savemodel({
    #     'epoch': 0,  # last epoch
    #     'state_dict': model.state_dict(),
    #     'best_prec1': 0.,
    #     'cfg': pruned_pkl['cfg']}, True, 499, 'trial07aiter', True)
    

    # class SingleInputNet(nn.Module):
    #     def __init__(self):
    #         super(SingleInputNet, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
    #         self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
    #         self.fc1 = nn.Linear(1440, 50)
    #         self.fc2 = nn.Linear(50, 10)
    #         self.size = 0

    #     def forward(self, x):
    #         x = self.conv1(x)
    #         print(x.size())
    #         self.size += x.numel()
    #         x = F.max_pool2d(x, 2)
    #         # self.size += x.numel()
    #         x = F.relu(x)
    #         # self.size += x.numel()
    #         x = self.conv2(x)
    #         self.size += x.numel()
    #         x = F.max_pool2d(x, 2)
    #         # self.size += x.numel()
    #         x = F.relu(x)
    #         # self.size += x.numel()
    #         x = x.view(-1, 1440)
    #         # self.size += x.numel()
    #         x = F.relu(self.fc1(x))
    #         # self.size += x.numel()
    #         x = self.fc2(x)
    #         self.size += x.numel()
    #         x = F.log_softmax(x, dim=1)
    #         # self.size += x.numel()
    #         return x
    
    # model = SingleInputNet()
    # summary(model, input_size=(3, 32, 32), device='cpu', batch_size=4)
    
    # model.size * 4/(1024**2)
