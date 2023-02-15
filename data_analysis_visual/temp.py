
f = open('/home/lianjun.wu/wulianjun/jiedizhuangzhi_crop2/ImageSets/Main/val.txt')

ids = []

for line in f:
    id = line.split('_')[0]
    if id not in ids: 
        ids.append(id)

from pdb import  set_trace
set_trace()
f.close()

f = open('/home/lianjun.wu/wulianjun/SG4/jiedizhuangzhi/ImageSets/Main/val_for_crop.txt','w')

for id in ids:
    f.write(id+'\n')

f.close()
