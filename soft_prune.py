#%%
# Working directory
import os
import argparse
import time
os.getcwd()
# Built in tools
import pickle
import itertools
from glob import glob  # find file path
import time
from collections import Counter, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
from torchvision import models
import torch.utils.data as Data

from load_birds import train_data_load, test_data_load
from load_cars import LoadCars
from load_aircraft import LoadAircrafts

from models import VGG, BCNN, BCNN_BN1D
from model_resnet import RESNET
import shutil
from utils import updateBN, savemodel

# print('torch version :' , torch.__version__)
# print('cuda available :' , torch.cuda.is_available())
# print('cudnn enabled :' , torch.backends.cudnn.enabled)

os.chdir('D:/')
os.getcwd()

#%%
parser = argparse.ArgumentParser(description='PyTorch Slimming main')
parser.add_argument('--pruned', default='', type=str, metavar='PATH',
                    help='path to pruned model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint or pruned model (default: none)')
parser.add_argument('--trial', default='', type=str,
                    help='training trial note (default: none)')
parser.add_argument('--epoch', type=int, default=50, metavar='N',
                    help='number of training epoch (default: 50)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--decay', type=float, default=1e-5, metavar='Decay',
                    help='weight decay (default: 1e-5)')
parser.add_argument('--sparsity', type=float, default=1e-4, metavar='SP',
                    help='penalize on bn factor (default: 1e-4)')
parser.add_argument('--first', type=float, default=1e-4, metavar='First',
                    help='penalize on first bn factor (default: 1e-4)')
parser.add_argument('--last', type=float, default=1e-4, metavar='Last',
                    help='penalize on last bn factor (default: 1e-4)')
parser.add_argument('--patience', type=int, default=30, metavar='P',
                    help='learning rate decay patience (default: 30)')
parser.add_argument('--freq', type=int, default=1, metavar='Freq',
                    help='freq of zero out parameters (default: 1)')
parser.add_argument('--percent', type=float, default=50,
                    help='scale sparse rate (default: 50)')  # set pruning rate
args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#%%
# All parameters setting
# dataset
dataset = 'Cars'
batch_size = 48
workders = 2
# model
pruned = args.pruned
resume = args.resume
num_classes = 196
cuda = True
# hyperparameters
lr = args.lr
decay = args.decay
epochs = args.epoch
channel_sparsity = True
sparsity_rate = args.sparsity
first = args.first
last = args.last
checkpoint_freq = 5
patience = args.patience
# trial id
trial = args.trial
# train
early_stop = False
soft = True
prune_freq = args.freq
percent = args.percent 
#%%
if __name__ == '__main__':
    print('Batch: {}, LR: {}, Sparsity: {}, Penalty: {}, Patience: {}, Pruned: {}, Freq: {}'.
          format(batch_size, lr, channel_sparsity, sparsity_rate, patience, percent, prune_freq))
    #%%
    # Seed
    # random.seed()
    torch.manual_seed(0)  # seed
    torch.backends.cudnn.benchmark = True
    # torch.initial_seed()  # 1
    # torch.cuda.initial_seed()  # 1

    #%%
    # Dataloader
    dataset = dataset
    BATCH = batch_size
    WORKERS = workders
    
    if dataset == 'Birds':
        data_path = 'C:/Dataset/Birds/'
        ROOT_TRAIN = data_path + 'train/'
        ROOT_TEST = data_path + 'test/'
        # dataloader
        label_train, train_loader = train_data_load(ROOT_TRAIN, BATCH, WORKERS)
        label_test, test_loader = test_data_load(ROOT_TEST, BATCH, WORKERS)
    elif dataset == 'Cars':
        cars = LoadCars()
        train_loader = cars.load_data(True, BATCH, WORKERS)
        test_loader = cars.load_data(False, BATCH, WORKERS)
    elif dataset == 'Aircrafts':
        aircrafts = LoadAircrafts()
        train_loader = aircrafts.createloader(dataset='train_valid', batch=BATCH, workers=WORKERS)
        test_loader = aircrafts.createloader(dataset='test', batch=BATCH, workers=WORKERS)
        

#%%
# Create a model
if __name__ == '__main__':
    # resume to the checkpoint if exists
    if not pruned:
        # net = BCNN(num_classes)
        net = BCNN(num_classes=num_classes, pretrained=True)
        if os.path.isfile(resume):
            print("Load the checkpoint")
            checkpoint = torch.load(resume, map_location='cpu')
            state_dict = OrderedDict([('classifier.weight', v) if k == 'fc.weight' else (k, v) for k, v in checkpoint['state_dict'].items()])
            state_dict = OrderedDict([('classifier.bias', v) if k == 'fc.bias' else (k, v) for k, v in state_dict.items()])
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(state_dict)
            print("Path:'{}'\nLast epoch: {} \nValidAcc: {:.2f}%"
                .format(resume, checkpoint['epoch'], best_prec1))
        else:
            print("No checkpoint found.".format(resume))            
    # fine-tune the model
    else:
        pruned_pkl = torch.load(pruned, map_location='cpu')
        net = BCNN(num_classes, cfg=pruned_pkl['cfg'])
        state_dict = OrderedDict([('classifier.weight', v) if k == 'fc.weight' else (k, v) for k, v in pruned_pkl['state_dict'].items()])
        state_dict = OrderedDict([('classifier.bias', v) if k == 'fc.bias' else (k, v) for k, v in state_dict.items()])
        net.load_state_dict(state_dict)
        print("Load pruned model: {}".format(pruned))

    #%%
    # send the model to the gpu
    if cuda:
        net.cuda()
        print('Use cuda: {}'.format(next(net.parameters()).is_cuda))  # check if model is in gpu

#%%
# Loss and Optimizer
if __name__ == '__main__':
    # Softmax is internally computed when using cross entropy.
    # Set parameters to be updated.
    LR = lr
    DECAY = decay
    EPOCHS = epochs

    # criterion = nn.NLLLoss()  
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss()=log_softmax() + NLLLoss() 
    optimizer = optim.SGD(net.parameters(),
                        lr=LR,
                        weight_decay=DECAY,
                        momentum=0.9,
                        nesterov=True)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-6)

    # CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-4)
    # ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=8, verbose=True, threshold=1e-4, min_lr=0.0001)
    # MultiStepLR(optimizer, milestones=[39, 59], gamma=0.1)
    # StepLR(optimizer, step_size=int(EPOCHS / 5), gamma=0.1)


    # if resume:
    #     print("Load the optimizer and lr-scheduler")
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])


#%%
# Train setting
if __name__ == '__main__':
    channel_sparsity = channel_sparsity
    checkpoint_freq = checkpoint_freq
    
    start_training = time.time()

    start_epoch = 0
    best_prec1 = 0.


#%%
# Train Model
if __name__ == '__main__':
    thre_list = []
    last_cfg = []
    print('Start training...')
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        start = time.time()
        # loss List
        list_loss_train = []
        list_loss_valid = []
        # training
        train_correct = 0
        train_total = 0
        net.train()  # activate autograd
        for i, (images, label) in enumerate(train_loader):
            # images, label = images, label
            images, label = images.cuda(), label.cuda()
            
            optimizer.zero_grad()  # clear buffer
            out = net(images) 
            train_loss = criterion(out, label)
            train_loss.backward()  
            # compute gamma gradient
            if channel_sparsity:
                updateBN(net, sparsity_rate, False, first, last)
            optimizer.step()  # update weights
            
            _, pred = torch.max(out.data, 1)  # max() return maximum and its index in each row
            train_total += float(label.size(0))
            train_correct += float((pred == label).sum())
        
        # soft filter prune
        if soft and (epoch+1) % prune_freq == 0:
            with torch.no_grad():
                # collect and sort gamma factors
                total = 0
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        total += m.weight.data.shape[0]  # number of gamma factors (as well as output channels) in all BN layers

                bn = torch.zeros(total)  # to store gamma factors later
                index = 0
                for m in net.modules():  # store all gamma factors
                    if isinstance(m, nn.BatchNorm2d):
                        size = m.weight.data.shape[0]  # number of gamma factors in a BN layer
                        bn[index:(index+size)] = m.weight.data.abs().clone()
                        index += size
                        
                # get pruning threshold
                percent = percent
                y, i = torch.sort(bn)  # return a tuple of (sorted_tensor, sorted_indices)
                thre_index = int(total * percent / 100)
                thre = y[thre_index]  # threshold
                thre_list.append(thre.item())
                # print('Channels pruned: {}%\nPruning threshold: {:.4f}'.format(percent, thre))
                
                # zero out the factors
                # print('Zero out the factors.')
                num_zero = 0  # store number of zero factors
                cfg = []  # store pruned model structure (number of output channels of each layer)
                cfg_mask = []  # store the pruning masks
                for k, m in enumerate(net.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        weight_copy = m.weight.data.clone()
                        mask = weight_copy.abs().gt(thre).float().cuda()  # mask to zero out the gamma factors under threshold
                        num_zero = num_zero + mask.shape[0] - torch.sum(mask)  # num of factors - num of 1 = num of 0
                        m.weight.data.mul_(mask)  # zero out
                        m.bias.data.mul_(mask)  # zero out
                        cfg.append(int(torch.sum(mask)))  # remain num of channel
                        cfg_mask.append(mask.clone())
                        # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        #     format(k, mask.shape[0], int(torch.sum(mask))))
                    elif isinstance(m, nn.MaxPool2d):
                        cfg.append('M')
            if cfg != last_cfg:
                print(cfg)
            last_cfg = cfg.copy()
        else:
            cfg = last_cfg.copy() or [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # validation
        valid_correct = 0
        valid_total = 0 
        net.eval()
        with torch.no_grad():
            for images, label in test_loader:
                # images, label = images, label
                images, label = images.cuda(), label.cuda()
                
                out = net(images)  # forward
                valid_loss = criterion(out, label)
                _, pred = torch.max(out.data, 1)  # max() return maximum and its index in each row
                valid_total += float(label.size(0))
                valid_correct += float((pred == label).sum())
        
        # time
        end = time.time()
        if epoch < 3:
            time_interval = (end - start)
            print('Time: {:.2f} secs'.format(time_interval))
            
        # metrics
        train_acc = 100*train_correct / train_total
        valid_acc = 100*valid_correct / valid_total
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        list_loss_train.append(train_loss)
        list_loss_valid.append(valid_loss)

        # learning rate scheme
        scheduler.step(valid_acc)
        
        # save model
        state = {
            'epoch': epoch,  # last epoch
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }
        suffix = trial
        if pruned:
            state['cfg'] = pruned_pkl['cfg']
            suffix += '_' + args.pruned.split('_')[-1][:-4]

        savemodel(state, is_best, checkpoint_freq, suffix, False)

        # print result
        if (epoch+1) % 1 == 0:

            print('Epoch:{}/{}\nAccuracy(Train/Valid):{:.02f}/{:.02f}% Loss(Train/Valid):{:.3f}/{:.3f}'.format(
                epoch, start_epoch + EPOCHS-1, train_acc, valid_acc, train_loss, valid_loss))

        # early stopping
        if early_stop and train_acc > 99.99:
            print('Early stop beacause train accuracy > 99.9.')
            break
        
    end_training = time.time()
    print('Time:', round((end_training - start_training)/60, 2), 'mins')

    pritn('Threshold:', thre_list)
# #%%
# # Prediction
# # Testing error
# correct = 0
# total = 0
# probs = []

# for i, (images, label) in enumerate(test_loader):
#     images, label = images, label
#     out = net.cpu()(images)
#     prob, predicted = torch.max(out.data, 1)  # max() return maximum and its index in each row
#     probs += prob.tolist()
#     total += label.size(0)
#     correct += (predicted == label).sum()
    
# print('Accuracy on test images: %d %%' % (100 * correct / total))



# #%%
# # Display predictions of images
# for i in range(1,10):
#     out = net(test_dataset.data[i])
#     prob, predicted = torch.max(out, 1)
#     plt.imshow(test_dataset.data[i].numpy(), cmap='gray')
#     plt.title('Prediction:%i   Probability:%.4f' % (test_dataset.targets[i].item(), prob.item()))
#     plt.show()

