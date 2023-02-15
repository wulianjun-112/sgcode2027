import os

with open('/data/wulianjun/datasets/SG5/导地线/ImageSets/Main/train_crop_no_fliter.txt','r') as f:
    aa = f.readlines()

with open('/data/wulianjun/datasets/SG5/导地线/ImageSets/Main/val_split.txt','r') as f:
    bb = f.readlines()

bb = [x.strip('\n') for x in bb]
train_split = []
val_split = []
for x in aa:
    
    xx = ''
    for i in x.split('_')[:-1]:
        xx += i+'_'
    xx = xx[:-1]
    if xx in bb:
        val_split.append(x)
    else:
        train_split.append(x)


with open('/data/wulianjun/datasets/SG5/导地线/ImageSets/Main/train_crop_split_no_fliter.txt','w') as f:
    for x in train_split:
        f.write(x)

with open('/data/wulianjun/datasets/SG5/导地线/ImageSets/Main/val_crop_split_no_fliter.txt','w') as f:
    for x in val_split:
        f.write(x)

from pdb import set_trace
set_trace()