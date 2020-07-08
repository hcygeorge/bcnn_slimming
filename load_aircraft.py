# Define how to load and process datasets
#%%
import os
import pickle
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import collections

import torch
import torchvision
from torchvision import datasets, transforms

#%%
# Custom Dataset
class Aircrafts(torch.utils.data.Dataset):
    """Create Aircraft dataloader."""
    
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
        # crop the banner of copyright  
        rect = 0, 0, image.size[0], image.size[1]-20
        image = image.crop(rect)
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
# Create Aircraft dataloader
class LoadAircrafts():
    """Create Aircraft dataloader(Pytorch)."""
    def __init__(self):
        self.data_path = 'C:/Dataset/Aircrafts'
        self.image_path = self.data_path + '/data/images'
        self.variant_path = self.data_path + '/data/variants.txt'
        self.train_anno_path = self.data_path + '/data/images_variant_train.txt'
        self.valid_anno_path = self.data_path + '/data/images_variant_val.txt'
        self.test_anno_path = self.data_path + '/data/images_variant_test.txt'
        self.encodelabel()
        self.train_paths, self.train_target = self.pathlist(self.train_anno_path)
        self.valid_paths, self.valid_target = self.pathlist(self.valid_anno_path)
        self.test_paths, self.test_target = self.pathlist(self.test_anno_path)
        self.imageprocess()

    def encodelabel(self):
        """Create dictionaries to encode and decode image labels to target(integer)."""
        with open(self.variant_path, 'r') as f:
            label_names = f.readlines()
        self.label2code = {}
        self.code2label = {}
        
        for idx, label in enumerate(label_names):
            self.label2code[label[:-1]] = idx
            self.code2label[idx] = label[:-1]
        
    def pathlist(self, images_varinat_txt):
        """Collect image paths and target values of Aircraft dataset.
        
        Parameters:
            images_varinat_txt (str): Path of document of image filenames and variants(labels).
        
        Returns:
            paths (list): List of image paths of the dataset.
            target (list): List of label code of the dataset.
        
        """
        with open(images_varinat_txt, 'r') as f:
            annos = f.readlines()
            
        paths = []
        labels = []
        for anno in annos:
            paths.append(self.image_path + '/' + anno[0:7] + '.jpg')
            labels.append(anno[8:-1])

        target = []
        for label in labels:
            target.append(self.label2code[label])
        
        # check if target start from 0
        if not min(target) == 0:
            print("Warning: Target doesn't start from 0.")
        
        return paths, target
        

    def imageprocess(self):
        """Define images preprocessing flow."""
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
            ])
        }

    def createloader(self, dataset='train', batch=32, workers=0, pin=True):
        """Create a Aircraft dataloader.
        
        Parameters:
            dataset (str): Input 'train', 'valid', 'train_valid' or 'test'.
            batch (int): Batch size of dataset.
        
        Returns:
            torch.utils.data.DataLoader
        
        """
        custom_dataset = {}
        custom_dataset['train'] = Aircrafts(self.train_paths,
                                            self.train_target,
                                            self.transform['train'])
        custom_dataset['valid'] = Aircrafts(self.valid_paths,
                                            self.valid_target,
                                            self.transform['test'])
        custom_dataset['train_valid'] = Aircrafts(self.train_paths + self.valid_paths,
                                                  self.train_target + self.valid_target,
                                                  self.transform['train'])
        custom_dataset['test'] = Aircrafts(self.test_paths,
                                           self.test_target,
                                           self.transform['test'])

        print("Dataset: {}\nSample Size: {}\nNumber of Class: {}".format(
            dataset, len(custom_dataset[dataset].data), len(self.label2code)))
        
        return torch.utils.data.DataLoader(custom_dataset[dataset],
                                           batch_size=batch,
                                           shuffle='train' in dataset,
                                           num_workers=workers,
                                           pin_memory=pin)

#%%
# unit test
if __name__ == '__main__':
    print('Unit test of load_aircraft.py')
    print('It should print 3 images with random flipping and rotation.')
    train = LoadAircrafts()
    trainloader = train.createloader(dataset='train_valid', batch=1)

    for idx, (image, target) in enumerate(trainloader):
        image = image.permute(2, 3, 1, 0)
        image = np.squeeze(image*0.5 + 0.5)

        plt.imshow(image)
        plt.title(train.code2label[target.item()])
        plt.show()
        if idx == 2:
            break
    