#%%
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
#%%

train_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(224, padding=4),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def train_data_load(root_train, batch, workers=2):
    """Return dict of class labels and train_loader."""
    train_dataset = datasets.ImageFolder(root_train, transform=train_transforms)
    CLASS = train_dataset.class_to_idx
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=workers)
    return CLASS, train_loader
 
def test_data_load(root_test, batch, workers=2):
    test_dataset = torchvision.datasets.ImageFolder(root_test,  transform=test_transforms)
    CLASS = test_dataset.class_to_idx
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=workers)
    return CLASS, test_loader

if __name__ == '__main___':
    # Set path
    path = 'C:/Dataset/CUB200/'
    ROOT_TRAIN = path + 'train/'
    ROOT_TEST = path + 'test/'

    # Create data loader
    SIZE_BATCH = 1
    label_train, train_loader = train_data_load(ROOT_TRAIN, SIZE_BATCH)
    label_test, test_loader = test_data_load(ROOT_TEST, SIZE_BATCH)


    for idx, (img, label) in enumerate(train_loader):
        img = np.squeeze(img)
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        if idx > 1:
            break


    # Compare the labels of training data and testing data
    label_train == label_test
    
    train_loader.dataset
    test_loader.dataset