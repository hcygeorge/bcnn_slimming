import imagesize
import os
from glob import glob
#%%
# Set path
path = 'C:/works/PythonCode/CUB_200_2011/data/'

path_images = path + 'images.txt'
path_split = path + 'train_test_split.txt'
train_save_path = path + 'train/'
test_save_path = path + 'test/'
 
image_path = 'C:/works/PythonCode/CUB_200_2011/data/images'

widths = []
heights = []
width, height = imagesize.get(image_path + "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
print(width, height)
widths.append(width)
heights.append(height)


#%%

class ImageStats():
    """Count number of images and their sizes in a folder recursively.
    
    Assume that the images with same category are stored in a folder.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.grand_total = 0
        self.class_total = {}
        self.widths = []
        self.heights = []
        self.get_file_paths(self.image_path)
        self.stats()
        print('Images Statistics\n----------------')
        print('Number of classes: {}'.format(self.num_classes))
        print('Number of images : {}'.format(self.grand_total))
        print('Average number of images (per class):{}'.format(self.average_num_images))
        
        print('\nAverage image size     :' + str(self.average_size))
        print('Range of width   :'+ str(self.width_range))
        print('Range of height  :'+ str(self.height_range))

        
    
    def get_file_paths(self, image_path):
        total = 0
        for path in glob(image_path + '/*', recursive=False):
            class_name = path.replace('\\', '/').split('/')[-2]
            if os.path.isdir(path):
                self.get_file_paths(path)
            else:
                filename = path.replace('\\', '/')
                self.widths.append(imagesize.get(filename)[0])
                self.heights.append(imagesize.get(filename)[1])
                total += 1
        self.class_total[class_name] = total
        self.grand_total += total

    def stats(self):
        self.num_images_per_class = [num for num in image_stats.class_total.values() if num != 0]
        self.num_classes = len(self.num_images_per_class)
        self.average_num_images = sum(self.num_images_per_class) / self.num_classes
        self.num_images_range = (round(min(self.num_images_per_class), 2), round(max(self.num_images_per_class), 2))
        
        self.average_size = (round(sum(self.widths)/self.grand_total, 2),
                             round(sum(self.heights)/self.grand_total, 2))
        self.width_range = (round(min(self.widths), 2), round(max(self.widths), 2))
        self.height_range = (round(min(self.heights), 2), round(max(self.heights), 2))
        

image_stats = ImageStats(image_path)


