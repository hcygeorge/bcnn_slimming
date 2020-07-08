
"""
Load images.txt for image paths
Load train_test_split.txt for train-test labels
"""
#%%
import os
import shutil
import numpy as np
import time
#%%
# Set path
path = 'C:/works/PythonCode/CUB_200_2011/data/'

path_images = path + 'images.txt'
path_split = path + 'train_test_split.txt'
train_save_path = path + 'train/'
test_save_path = path + 'test/'
 
#%%
# Collect image paths
images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(line.strip('\n').split(',')[0])

#%%
# Collect train-test labels
split = []
with open(path_split, 'r') as f:
    for line in f:
        split.append(line.strip('\n').split(',')[0])


num_train = 0
num_test = 0
for label in split:
    if int(label.split(' ')[-1]):
        num_train += 1
    else:
        num_test += 1
print('Size of dataset\ntrain: {:d}\ntest : {:d}'.format(num_train, num_test)) 


#%%
# Split training and testing data
time_start = time.time()

num_images = len(images)
for i in range(num_images):
    filepath = images[i].split(' ')[1]
    folder = filepath.split('/')[0]
    is_train = int(split[i][-1])  # belongs to training data or not
    # copy the image to training set
    if is_train:
        if os.path.isdir(train_save_path + folder):
            shutil.copy(path + 'images/' + filepath, train_save_path + filepath)
        else:
            os.makedirs(train_save_path + folder)
            shutil.copy(path + 'images/' + filepath, train_save_path + filepath)
    # copy the image to testing set
    else:
        if os.path.isdir(test_save_path + folder):
            shutil.copy(path + 'images/' + filepath, test_save_path + filepath)
        else:
            os.makedirs(test_save_path + folder)
            shutil.copy(path + 'images/' + filepath, test_save_path + filepath)
    if (i+1) % 100 == 0:
        print('Process: [{}/{}]'.format(i+1, num_images))
    elif i+1 == num_images:
        print('Process: [{}/{}], done.'.format(i+1, num_images))

time_end = time.time()
print('Time: {:.2f} secs'.format((time_end - time_start)))
