#coding=utf-8
import os
import os.path as osp
import shutil

all = ['导地线','附属设施','杆塔','基础','绝缘子','通道环境','小金具','大金具', '接地装置']

data_root = '/data/wulianjun/datasets/SG5'

for data_brance in all:
    file_names = os.listdir(osp.join(data_root,data_brance))
    if not osp.exists(osp.join(data_root,data_brance,'JPEGImages')):
        os.mkdir(osp.join(data_root,data_brance,'JPEGImages'))
    if not osp.exists(osp.join(data_root,data_brance,'Annotations')):
        os.mkdir(osp.join(data_root,data_brance,'Annotations'))

    for file_name in file_names:
        if file_name.endswith('.jpg'):
            shutil.move(osp.join(data_root,data_brance,file_name),osp.join(data_root,data_brance,'JPEGImages',file_name))
        elif file_name.endswith('.xml'):
            shutil.move(osp.join(data_root,data_brance,file_name),osp.join(data_root,data_brance,'Annotations',file_name))
        else:
            from pdb import set_trace
            set_trace()
        
## for valset

# data_root = '/data1/wlj/StateGridv5/验证集'
# subroots = os.listdir(data_root)

# if not osp.exists(osp.join(data_root,'JPEGImages')):
#     os.mkdir(osp.join(data_root,'JPEGImages'))
# if not osp.exists(osp.join(data_root,'Annotations')):
#     os.mkdir(osp.join(data_root,'Annotations'))

# for subroot in subroots:
#     file_names = os.listdir(osp.join(data_root,subroot))
#     for file_name in file_names:
#         if file_name.endswith('.JPG'):
#             os.rename(osp.join(data_root,subroot,file_name),osp.join(data_root,subroot,file_name.replace('.JPG','.jpg')))
#             shutil.move(osp.join(data_root,subroot,file_name.replace('.JPG','.jpg')),osp.join(data_root,'JPEGImages',file_name.replace('.JPG','.jpg')))
#         elif file_name.endswith('.jpg'):
#             shutil.move(osp.join(data_root,subroot,file_name),osp.join(data_root,'JPEGImages',file_name))

#         elif file_name.endswith('.xml'):
#             shutil.move(osp.join(data_root,subroot,file_name),osp.join(data_root,'Annotations',file_name))
        
#         else:
#             from pdb import set_trace
#             set_trace()