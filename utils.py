import os
import time
import torch
import torch.nn as nn
#%%
# Sub subgradient descent for L1-norm
def updateBN(model, scale=1e-4, verbose=False, fisrt=1e-4, last=1e-4):
    """Subgradient descent for L1-norm.
    
    Args:
        model, nn.modules
        coef, scaling factor of L1 penalty term
    """
    for idx, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # if idx == 'features.28':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            # if idx == 'features.35':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            if idx == 'features.41':
                m.weight.grad.data.add_(last*torch.sign(m.weight.data))
            else:
                m.weight.grad.data.add_(scale*torch.sign(m.weight.data))  # L1



#%%
def savemodel(state, is_best, freq=10, suffix='', verbose=False):
    serial_number = time.strftime("%m%d")
    checkpoint = './model/checkpoint{}_{:s}.pkl'.format(serial_number, suffix)
    bestmodel = './model/bestmodel{}_{:s}.pkl'.format(serial_number, suffix)
        
    if not os.path.exists('./model'):
        os.makedirs('./model')
        print("Create a folder 'model' under working directory.")
        
    if is_best:
        torch.save(state, bestmodel)
    elif (state['epoch'] + 1) % freq == 0:
        torch.save(state, checkpoint)
        print('Model saved.')
        
    if verbose:
        print('Filepaths: {:s}/{:s}'.format(bestmodel, checkpoint))
        
