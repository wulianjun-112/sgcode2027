
import os 
from pdb import set_trace
import cv2
ff = open('/data/wulianjun/StateGridv5/附属设施/ImageSets/Main/train.txt')

for line in ff:
    img_name = line.split('\n')[0]+'.jpg'
    img = cv2.imread('/data/wulianjun/StateGridv5/附属设施/JPEGImages/'+img_name)
    print(img.shape)
    set_trace()
    