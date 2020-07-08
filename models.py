import torch
import torch.nn as nn
import torchvision
import math  # init
from sqrtm import sqrtm
#%%
class VGG(nn.Module):
    def __init__(self, dataset=None, num_classes=None, pretrained=True, cfg=None):
        super(VGG, self).__init__()
        
        # check if dataset is cifar100 or cifar10
        if dataset == 'cifar100':
            num_classes = 100
            out_size = 512*1*1
        elif dataset == 'cifar10':
            num_classes = 10
            out_size = 512*1*1
        elif dataset == 'cub200':
            num_classes = 200
            out_size = 512*7*7
        else:
            out_size = 512*7*7

        # define model structure
        if pretrained:
            print('Use pretrained VGG feature extractor')
            self.feature = torchvision.models.vgg16(pretrained=True).features
            self.feature = nn.Sequential(*list(self.feature.children()))  # Remove pool5
            self.classifier = nn.Linear(out_size, num_classes)
        else:
            if cfg is None:
                cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

            self.feature = self.make_layers(cfg, batch_norm=False)
            self.classifier = nn.Linear(out_size, num_classes)
            self._initialize_weights()



    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        # x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        print('Initial model parameters...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class BCNNResnet(nn.Module):
    """Mean field B-CNN model.

    input 3*448*448, output 512*28*28
    input 3*224*224, output 512*14*14 (ResNet34 or less)

    Attributes:
        features: features extractor
        nodes: number of output channels of features extractor
        classifier: fc
        relu5_3: activation between features and fc
    """
    def __init__(self, num_classes, pretrained=True, cfg=None):
        """Declare all needed layers.

        Args:
            num_classes, int.
        """
        super(BCNNResnet, self).__init__()
        # define model structure
        
        self.out_channel = 512  # feature output channels
        self.out_size = 14*14  # feature output size of Resnet[:-2]
        
        if pretrained and not cfg:
            self.features = torchvision.models.resnet34(pretrained=pretrained)
            print('Create pretrained ResNet34 model.')
        elif not pretrained and not cfg:
            self.features = torchvision.models.resnet34(pretrained=False).features
            self.apply(BCNN._initParameter)  # initialize model parameters
            print('Initialize ResNet34 model parameters.')
            
        self.features = torch.nn.Sequential(*list(self.features.children())[:-2])  # Remove Avgpool and Linear
        
        # Mean field pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False) 

        # Classification layer
        self.classifier = torch.nn.Linear(in_features=self.out_channel**2,
                                          out_features=num_classes,
                                          bias=True)
        
    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input
        N = X.size()[0]  # store batch size
        X = self.features(X)
        X = self.relu5_3(X)
        # Classical bilinear pooling
        X = torch.reshape(X, (N, self.out_channel, self.out_size))
        X = torch.bmm(X, X.permute(0, 2, 1))  # bilinear pooling
        X = torch.div(X, self.out_size)    
        X = torch.reshape(X, (N, self.out_channel**2))
        # Normalization
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # Classification
        # print(X.size())
        X = self.classifier(X)
        return X
    

class BCNN(nn.Module):
    """Mean field B-CNN model.

    input 3*448*448, output 512*28*28
    input 3*224*224, output 512*14*14

    Attributes:
        features: features extractor
        nodes: number of output channels of features extractor
        fc: fc
        relu5_3: activation between features and fc
    """
    def __init__(self, num_classes, pretrained=True, cfg=None, bn=True):
        """Declare all needed layers.

        Args:
            num_classes, int.
        """
        super(BCNN, self).__init__()
        # define model structure
        
        self.out_channel = 512  # feature output channels
        self.out_size = 14*14  # feature output size
        
        if pretrained and bn and not cfg:
            self.features = torchvision.models.vgg16_bn(pretrained=pretrained).features
            print('Create pretrained model with BN layer.')
        elif pretrained and not bn and not cfg:
            self.features = torchvision.models.vgg16(pretrained=pretrained).features
            print('Create pretrained model without BN layer.')
        elif cfg:
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channelss
            print('Create model backbone by cfg.')
        else:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channels
            print('Create model backbone of VGG16 structure.')
            

        self.features = torch.nn.Sequential(*list(self.features.children())[:-2])  # Remove pool5
        # Mean field pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False) 

        # Classification layer
        # print('Last conv layer number of filters: ', self.out_channel)
        self.classifier = torch.nn.Linear(in_features=self.out_channel**2,
                                          out_features=num_classes,
                                          bias=True)
        if not pretrained:
            self.apply(BCNN._initParameter)  # initialize model parameters
            print('Initialize model parameters.')


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    # print('Add batchnorm after conv2d.')
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input
        N = X.size()[0]  # store batch size
        X = self.features(X)
        X = self.relu5_3(X)
        # print(X.size())
        # Classical bilinear pooling
        X = torch.reshape(X, (N, self.out_channel, self.out_size))
        # print(X.size())
        X = torch.bmm(X, X.permute(0, 2, 1))  # bilinear pooling
        X = torch.div(X, self.out_size)    
        # print('Size of features outer-product: ', X.size())
        X = torch.reshape(X, (N, self.out_channel**2))
        # Normalization
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # Classification
        # print(X.size())
        X = self.classifier(X)
        return X
    

class BCNN_BN1D(nn.Module):
    """Mean field B-CNN model, add batchnorm2d after bilinear pooling.

    input 3*448*448, output 512*28*28
    input 3*224*224, output 512*14*14

    Attributes:
        features: features extractor
        nodes: number of output channels of features extractor
        fc: fc
        relu5_3: activation between features and fc
    """
    def __init__(self, num_classes, pretrained=True, cfg=None, bn=True):
        """Declare all needed layers.

        Args:
            num_classes, int.
        """
        super(BCNN_BN1D, self).__init__()
        # define model structure
        
        self.out_channel = 512  # feature output channels
        self.out_size = 14*14  # feature output size
        
        if pretrained and bn and not cfg:
            self.features = torchvision.models.vgg16_bn(pretrained=pretrained).features
            print('Create pretrained model with BN layer.')
        elif pretrained and not bn and not cfg:
            self.features = torchvision.models.vgg16(pretrained=pretrained).features
            print('Create pretrained model without BN layer.')
        elif cfg:
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channelss
            print('Create model backbone by cfg.')
        else:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channels
            print('Create model backbone of VGG16 structure.')
            

        self.features = torch.nn.Sequential(*list(self.features.children())[:-2])  # Remove pool5
        # Mean field pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False) 

        # Classification layer
        # print('Last conv layer number of filters: ', self.out_channel)
        self.classifier = torch.nn.Linear(in_features=self.out_channel**2,
                                          out_features=num_classes,
                                          bias=True)
        self.bn1d = nn.BatchNorm1d(512**2)
        
        if not pretrained:
            self.apply(BCNN._initParameter)  # initialize model parameters
            print('Initialize model parameters.')


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    # print('Add batchnorm after conv2d.')
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input
        N = X.size()[0]  # store batch size
        X = self.features(X)
        X = self.relu5_3(X)
        # print(X.size())
        # Classical bilinear pooling
        X = torch.reshape(X, (N, self.out_channel, self.out_size))
        # print(X.size())
        X = torch.bmm(X, X.permute(0, 2, 1))  # bilinear pooling
        X = torch.div(X, self.out_size)    
        # print('Size of features outer-product: ', X.size())
        X = torch.reshape(X, (N, self.out_channel**2))
        # Normalization
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        
        X = self.bn1d(X)
        # Classification
        # print(X.size())
        X = self.classifier(X)
        return X

class ImproveBCNN(nn.Module):
    """Mean field B-CNN model.

    input 3*448*448, output 512*28*28
    input 3*224*224, output 512*14*14

    Attributes:
        features: features extractor
        nodes: number of output channels of features extractor
        fc: fc
        relu5_3: activation between features and fc
    """
    def __init__(self, num_classes, pretrained=True, cfg=None, bn=True):
        """Declare all needed layers.

        Args:
            num_classes, int.
        """
        super(ImproveBCNN, self).__init__()
        # define model structure
        
        self.out_channel = 512  # feature output channels
        self.out_size = 14*14  # feature output size
        
        if pretrained and bn and not cfg:
            self.features = torchvision.models.vgg16_bn(pretrained=pretrained).features
            print('Create pretrained model with BN layer.')
        elif pretrained and not bn and not cfg:
            self.features = torchvision.models.vgg16(pretrained=pretrained).features
            print('Create pretrained model without BN layer.')
        elif cfg:
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channelss
            print('Create model backbone by cfg.')
        else:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            self.features = self.make_layers(cfg, bn)
            self.out_channel = cfg[-2]  # feature output channels
            print('Create model backbone of VGG16 structure.')
            

        self.features = torch.nn.Sequential(*list(self.features.children())[:-2])  # Remove pool5
        # Mean field pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False)

        # Classification layer
        self.fc = torch.nn.Linear(in_features=self.out_channel**2,
                                          out_features=num_classes,
                                          bias=True)
        if not pretrained:
            self.apply(BCNN._initParameter)  # initialize model parameters
            print('Initialize model parameters.')


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    # print('Add batchnorm after conv2d.')
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
                

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input
        N = X.size()[0]  # store batch size
        X = self.features(X)
        X = self.relu5_3(X)
        # Classical bilinear pooling
        X = torch.reshape(X, (N, self.out_channel, self.out_size))
        X = torch.bmm(X, X.permute(0, 2, 1))  # bilinear pooling  N*512*196 x N*196*512
        X = torch.div(X, self.out_size)    
        # Sqrt of a matrix, add small number to diagonal to make it positive definite
        X = sqrtm(X + torch.eye(512))
        X = torch.reshape(X, (N, self.out_channel**2))
        # Normalization
        X = torch.sign(X) * torch.sqrt(torch.abs(X))
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # Classification
        # print(X.size())
        X = self.fc(X)
        return X
    

class BCNNMultiModel(nn.Module):
    """Mean field B-CNN model.

    The model accepts a 3*448*448 input, and the relu5-3 activation has shape
    512*28*28 if using VGG16 structure.

    Attributes:
        features: features extractor
        nodes: number of output channels of features extractor
        fc: fc
        relu5_3: activation between features and fc
    """
    def __init__(self, num_classes, bn=True):
        """Declare all needed layers.

        Args:
            num_classes, int.
        """
        super(BCNNMultiModel, self).__init__()
        # define model structure
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features1 = self.make_layers(cfg, bn)
        self.features2 = self.make_layers(cfg, bn)
        self.out_channel = cfg[-2]  # feature output channelss
        self.out_size = 14*14  # feature output size
        self.features1 = torch.nn.Sequential(*list(self.features1.children())[:-2])  # Remove pool5
        self.features2 = torch.nn.Sequential(*list(self.features2.children())[:-2])  # Remove pool5
        self.relu5_31 = torch.nn.ReLU(inplace=True)
        self.relu5_32 = torch.nn.ReLU(inplace=True) 
        self.fc = torch.nn.Linear(in_features=self.out_channel**2,
                                          out_features=num_classes,
                                          bias=True)
        self.apply(BCNN._initParameter)  # initialize model parameters

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input
        N = X.size()[0]  # store batch size

        X1 = self.features1(X)
        X2 = self.features2(X)
        
        X1 = self.relu5_31(X1)
        X2 = self.relu5_32(X2)
        # Classical bilinear pooling
        X1 = torch.reshape(X1, (N, self.out_channel, self.out_size))
        X2 = torch.reshape(X2, (N, self.out_channel, self.out_size))
        X = torch.bmm(X1, X2.permute(0, 2, 1))  # bilinear pooling
        X = torch.div(X, self.out_size)   
        X = torch.reshape(X, (N, self.out_channel**2))
        # Normalization
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8)
        X = torch.nn.functional.normalize(X, p=2, dim=1)

        # print(X.size())
        # Classification
        X = self.fc(X)
        return X

if __name__ == '__main__':
    # check model output size


    net = BCNN(200, True)
    x = torch.FloatTensor(1, 3, 448, 448)
    y = net(x)  # 512*14*14
    print(y.data.shape)

    
    x = torch.FloatTensor(3, 512, 196)
    x = torch.bmm(x, x.permute(0, 2, 1))  # bilinear pooling
    x = torch.div(x, 196) 
    bn2 = nn.BatchNorm2d(512)
    x = torch.reshape(x, (3, 512, 1, 512))
    y = bn2(x)
    x = torch.reshape(x, (3, 512**2))
    x.size()
    y.size()
    
    x = torch.FloatTensor(12, 512*512)
    net.fc(x.cuda())
    
    
    net = BCNN(200)
    for idx, m0 in net.named_modules():
        print(idx)
        print(idx == 'features.41')

    # test
    # import torchvision
    # a = torchvision.models.resnet34(pretrained=True)
    # b = torch.nn.Sequential(*list(a.children())[:-2])
    # b(torch.randn((1, 3, 224, 224))).size()
    x = torch.FloatTensor(16, 512, 196)
    bn1 = nn.BatchNorm1d(512**2)
    y = bn1(x)
    
    for name, para in bn1.named_parameters():
        print(name, para.size())
    