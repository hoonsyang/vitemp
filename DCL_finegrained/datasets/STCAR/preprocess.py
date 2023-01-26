import os
import sys
from glob import glob

import cv2

root = '/my/path/to//StanfordCars/'
cars_train_dataroot = os.path.join(root, 'cars_train')
cars_test_dataroot = os.path.join(root, 'cars_test')
savepath = os.path.join(root, 'cars')

count = 1
files = glob(cars_train_dataroot+'*/*.jpg', recursive=True)
files.extend(glob(cars_test_dataroot+'*/*.jpg', recursive=True))
for imgfile in files:
    path = imgfile.split('/')[:-2]
    filename = imgfile.split('/')[-1]
    n = cv2.imread(imgfile)    
    newpath = os.path.join(savepath, f'{str(count).zfill(5)}.jpg')
    cv2.imwrite(newpath, n)
    count += 1
print(f'Total: {count} images saved to: {savepath}')
    