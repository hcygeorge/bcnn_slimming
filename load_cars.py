# Define how to load and process datasets
#%%
import os
import pickle
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspecs

import torch
import torchvision
from torchvision import datasets, transforms

#%%
# Custom Dataset
class Cars(torch.utils.data.Dataset):
    """Create Cars dataloader."""
    
    def __init__(self, img_list, label_list,
                 transform=None, target_transform=None):
        """Load the dataset.
        Args
            img_list, str: List of images
            label_list, list: List of labels
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
        """
        self.data = img_list
        self.label = label_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        # load One sample in a time
        image, target = self.data[index], self.label[index]
        image = Image.open(image)
        # check if it is a grayscale image
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        # preprocessing data if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        # return x and y
        return image, target

    def __len__(self):
        """Length of the dataset."""
        return len(self.data)



#%%
# Create Standford Cars dataloader
class LoadCars():
    """Create Standford Cars dataloader(Pytorch)."""
    def __init__(self):
        # Path of data
        self.data_path = 'C:/Dataset/StandfordCar'
        self.label_path = os.path.join(self.data_path, 'devkit/cars_meta.mat')
        self.train_path = self.data_path + '/cars_train'
        self.test_path = self.data_path + '/cars_test'
        self.train_anno_path = os.path.join(self.data_path, 'devkit/cars_train_annos.mat')
        self.test_anno_path = os.path.join(self.data_path, 'devkit/cars_test_annos_withlabels.mat')
        # Create dataloaders
        self.path_list()
        self.transform()
        print('Dataset Path: {:s}\nNumber of Class: {}\nTrain Data: {}\nTest Data: {}'.format(
            self.data_path, len(np.unique(self.train_label)), len(self.train_label), len(self.test_data)))

        
    def path_list(self):
        # Load image paths and labels
        label_name = scipy.io.loadmat(self.label_path)
        label_names = (label_name['class_names'][0])

        train_anno = scipy.io.loadmat(self.train_anno_path)['annotations'][0]
        test_anno = scipy.io.loadmat(self.test_anno_path)['annotations'][0]

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        for i in range(len(train_anno)):
            self.train_data.append(os.path.join(self.train_path, train_anno[i][5][0]))
            self.train_label.append(int(train_anno[i][4]) - 1)
            
        for i in range(len(test_anno)):
            self.test_data.append(os.path.join(self.test_path, test_anno[i][5][0]))
            self.test_label.append(int(test_anno[i][4]) - 1)
            
        # Check labels (Pytorch require)
        if min(self.train_label) == 0 and min(self.test_label) == 0:
            print('Check if label start from 0: True')

    def transform(self):
        # Transform
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

    def load_data(self, train=True, batch=32, workers=0, pin=True):
        train_dataset = Cars(self.train_data,
                             self.train_label,
                             self.transform['train'])
        test_dataset = Cars(self.test_data,
                            self.test_label,
                            self.transform['test'])
        
        # DataLoader
        if train == True:
            return torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=pin)
        else:
            return torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch,
                                               shuffle=False,
                                               num_workers=workers,
                                               pin_memory=pin)

#%%
# function test
# if __name__ == '__main__':
#     cars = LoadCars()
#     train_loader = cars.load_data(True, batch=2)
    

#     plt.figure(figsize = (20, 20))

#     for idx, (img, label) in enumerate(train_loader):
#         # print(img.size())
#         plt.subplot(1, 5, idx+1)
#         plt.axis('off') 
#         plt.imshow(img[0].permute(1, 2, 0).numpy(), vmin=-0.5, vmax=0.5)
#         plt.subplots_adjust(wspace=0, hspace=0)
#         plt.savefig('./plot/data_aug.pdf', bbox_inches = 'tight',
#                     pad_inches = 0, format='pdf')
        
#         # plt.show()
#         if idx == 4:
#             break




