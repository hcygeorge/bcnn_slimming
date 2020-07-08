# summary.py
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models import BCNN
#%%
# model_path = 'C:/works/PythonCode/CUB_200_2011/model/sparsity1e-5/checkpoint0407_e100_acc73pruned.pkl'
# pruned = torch.load(model_path, map_location='cpu').cuda()
# pruned.nodes = 510
net = BCNN(200).cuda()



summary(net, input_size=(3, 224, 224))

# summary(pruned, input_size=(3, 448, 448))
# summary(newmodel, input_size=(3, 448, 448))


newmodel.fc.weight.size()
newmodel.fc.weight.data.dtype
#%%
# Input size (MB): 2.30
# Forward/backward pass size (MB): 700.86
# Params size (MB): 209.74
# Estimated Total Size (MB): 912.89

# Input size (MB): 2.30
# Forward/backward pass size (MB): 1286.25
# Params size (MB): 256.17
# Estimated Total Size (MB): 1544.71

(3*448*448) * (32/8) / 1024**2  # 2.29 MB

import sys
# a = torch.randn((3, 448, 448))
# a.dtype
# a.size()
# a.storage()
# sys.getsizeof(a.storage()) / 1024**2  # 2.2969 MB

# sys.getsizeof()
# a.element_size()
