#%%
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import VGG, BCNN
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from load_birds import train_data_load, test_data_load
from load_cars import LoadCars
from load_aircraft import LoadAircrafts
from models import VGG, BCNN, BCNN_BN1D
from utils import savemodel

os.chdir('D:/')
os.getcwd()

#%%
parser = argparse.ArgumentParser(description='PyTorch Slimming prune')
# parser.add_argument('--dataset', type=str, default='cifar10',
#                     help='training dataset (default: cifar10)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
parser.add_argument('--percent', type=float, default=50,
                    help='scale sparse rate (default: 50)')  # set pruning rate

parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
# parser.add_argument('--save', default='', type=str, metavar='PATH',
#                     help='path to save prune model (default: none)')
parser.add_argument('--plot', action='store_true', default=False,
                    help='path to save prune model (default: none)')
parser.add_argument('--trial', default='', type=str,
                    help='training trial note (default: none)')
args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

#%%
# Prune settings
print('Start pruning...')
# dataset
dataset = 'Cars'
batch_size = 48
workers = 0
# model
model_path = args.model
data_class = {'Aircrafts': 100, 'Birds': 200, 'Cars': 196}
num_classes = data_class[dataset]
# pruned ratio
percent = args.percent 
# plot
fig = 'png'
plotting = args.plot
# trial id
trial = args.trial
test = False

#%%
# Load trained model
# args.model <> model_path
# args.start_epoch <> start_epoch
if model_path:
    if os.path.isfile(model_path):
        model_pkl = torch.load(model_path)
        # Correct the name of fully connected layer
        state_dict = OrderedDict([('classifier.weight', v) if k == 'fc.weight' else (k, v) for k, v in model_pkl['state_dict'].items()])
        state_dict = OrderedDict([('classifier.bias', v) if k == 'fc.bias' else (k, v) for k, v in state_dict.items()])
        if 'cfg' in model_pkl.keys():
            model = BCNN(num_classes, cfg=model_pkl['cfg'])
        else:
            model = BCNN(num_classes)
        model.load_state_dict(state_dict)
        print("Loaded trained model: '{}'".format(model_path))
        model.cuda()
    else:
        print("No model: ".format(model_path))

#%%
# Collect and sort gamma factors
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]  # number of gamma factors (as well as output channels) in all BN layers

bn = torch.zeros(total)  # to store gamma factors later
index = 0
for m in model.modules():  # store all gamma factors
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]  # number of gamma factors in a BN layer
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
print('Total Gamma Parameters: {}'.format(len(bn)))
#%%
# Get pruning threshold
percent = percent
y, i = torch.sort(bn)  # return a tuple of (sorted_tensor, sorted_indices)
thre_index = int(total * percent / 100)
thre = y[thre_index]  # threshold

print('Channels pruned: {}%\nPruning threshold: {:.4f}'.format(percent, thre))
#%%
# Print weight histogram
if plotting:
    plt.grid()
    n, bins, patches = plt.hist(bn,
                                bins=128,
                                color='lightblue',
                                linewidth=5)
    plt.title(dataset, fontsize=16)
    plt.xlabel(r'$\gamma$', fontsize=14)
    plt.ylabel('Frequency', fontsize=16)
    patches[0].set_fc('black')
    plt.savefig('./plot/trial{}{:.0f}pdf.{}'.format(
        model_path.split('trial')[-1][:-4], percent, fig), format=fig)
    plt.savefig('./plot/trial{}{:.0f}pdf.{}'.format(
        model_path.split('trial')[-1][:-4], percent, 'eps'), format='eps')
    plt.tight_layout()
    plt.show()

    plt.grid()
    n, bins, patches = plt.hist(bn,
                                bins=128,
                                linewidth=5,
                                color="lightblue",
                                cumulative=True,
                                density=True)
    plt.title(dataset, fontsize=16)
    plt.xlabel(r'$\gamma$', fontsize=16)
    plt.ylabel('CDF', fontsize=14)
    # plt.axvline(thre, color='k', linestyle='dashed', linewidth=1.5, label='{}% pruned'.format(percent*100))
    # plt.legend(loc='lower right')
    plt.savefig('./plot/trial{}{:.0f}cdf.{}'.format(
        model_path.split('trial')[-1][:-4], percent, fig), format=fig)
    plt.savefig('./plot/trial{}{:.0f}cdf.{}'.format(
        model_path.split('trial')[-1][:-4], percent, 'eps'), format='eps')
    plt.tight_layout()
    plt.show()

#%%
# Dataloader
dataset = dataset
BATCH = batch_size
WORKERS = workers

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
# Test before prune
# args.cuda <> cuda
cuda = True

def testmodel(model):
    with torch.no_grad():
        correct = 0
        # if __name__ == '__main__':
        model.eval()
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            # return correct / float(len(test_loader.dataset))

if test:
    print('Test model before pruning.')
    testmodel(model)
#%%
# Zero out the factors
print('Zero out the factors.')
pruned = 0  # store number of zero factors
cfg = []  # store pruned model structure (number of output channels of each layer)
cfg_mask = []  # store the pruning masks
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()  # mask to zero out the gamma factors under threshold
        pruned = pruned + mask.shape[0] - torch.sum(mask)  # num of factors - num of 1 = num of 0
        m.weight.data.mul_(mask)  # zero out
        m.bias.data.mul_(mask)  # zero out
        cfg.append(int(torch.sum(mask)))  # remain num of channel
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')




#%%
# Make real prune
print('Make real prune...')
cfg.append('M')
newmodel = BCNN(num_classes, pretrained=False, cfg=cfg)  # create pruned model backbone
newmodel.cuda()

layer_id_in_cfg = 0
start_mask = torch.ones(3)

end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # find non-zero indices of factors
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
        w = m0.weight.data[:, idx0, :, :].clone()
        w = w[idx1, :, :, :].clone()
        m1.weight.data = w.clone()
        m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        # create a prune-sized output channels of last batch norm layer
        if 'cfg' in model_pkl.keys():
            pruned_channel = model_pkl['cfg'][-2]
        else:
            pruned_channel = 512
        channels = torch.ones(pruned_channel, 14, 14)
        # zero out the channel by idx0
        for i in range(pruned_channel):
            if i not in idx0:
                channels[i,:,:] = torch.zeros(channels[i,:,:].size())
        # show non zero channel index before outer product      
        # for i in range(512):
        #     if channels[i].sum() == 0:
        #         print(i)
        channels = torch.reshape(channels, (1, pruned_channel, 14 * 14))
        outer = torch.bmm(channels, channels.permute(0, 2, 1))
        flatten = outer.reshape(-1, pruned_channel**2)
        # collect non-zero column idx after outer product and reshape
        idxs = []
        for i in range(pruned_channel**2):
            if flatten[:, i].item() != 0:
                idxs.append(i)
        # transfer model weights to newmodel
        m1.weight.data = m0.weight.data[:, idxs].clone()


#%%
# Save pruned model
print('Saving model...')
# args.save <> save

# save_path = '{}_pruned{}.pkl'.format(model_path[:-4], int(percent*100))
# torch.save({'cfg': cfg, 'model': newmodel.state_dict()}, save_path)
# print('Pruned model: {}'.format(save_path))

# checkpoint
# notes = model_path.split('/')[-1][:-4] + '_pruned{}.pkl'.format(percent)
suffix = trial + '_pruned{:.0f}'.format(percent)
    
savemodel({
    'epoch': 0,  # last epoch
    'state_dict': newmodel.state_dict(),
    'best_prec1': 0.,
    'cfg': cfg}, True, 499, suffix, True)


print('Finish pruning.\n')
#%%
# test
# channels = torch.ones(512, 28, 28)
# idxs = [44, 122]
# # prune channels 44, 122
# for idx in range(512):
#     if idx in idxs:
#         channels[idx, :] = torch.zeros_like(channels[idx, :])


# channels = torch.reshape(channels, (1, 512, 28 * 28))
# channels.size()
# outer = torch.bmm(channels, channels.permute(0, 2, 1))
# outer.size()

# new = outer.reshape(-1, 512**2)
# new.size()
# idxs = []
# for i in range(262144):
#     if new[:,i].item() != 0:
#         idxs.append(i)
# len(idxs)
# new[:,44].item() == 0
# a = np.argwhere(new)

# a.size()
# # check
# for i in range(512):
#     # row == torch.zeros_like(row)
#     if torch.sum(new[i, :]) == 0:
        
#         print(new[i, :].size())
#         print(i)

# for i, row in enumerate(outer):
#     # row == torch.zeros_like(row)
#     if sum(row) == 0:
#         print(i)

# idxs = []
# for i in range(512):
#     idxs.add(44 + i*512)
#     idxs.append(122 + i*512)

# len(idxs)

# 512*512 - 510*510