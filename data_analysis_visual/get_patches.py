import numpy as np
import cv2
import os
from tqdm import tqdm 
import xml.etree.ElementTree as ET
patches = 16
files = os.listdir('/data/wulianjun/StateGridv5/小金具/Annotations')
xml_root = '/data/wulianjun/StateGridv5/小金具/Annotations/'
CLASSES = ['螺母锈蚀', '缺销', '销钉安装不到位', '螺母安装不规范', 
'缺螺母', '缺垫片', '缺螺栓', '垫片锈蚀']
target_root = '/data/wulianjun/StateGridv5/xiaojinju_patch/'
all_xml_file = []
def readXml(xmlfile):

    DomTree = ET.parse(xml_root+xmlfile)
    root = DomTree.getroot()
    bboxes = []
    labels = []
    for objects in root.findall('object'):
        class_label = objects.find('name').text
        if class_label in CLASSES :
            # if class_label in classNames:
            bnd_box = objects.find('bndbox')
            bbox = [
                        int(float(bnd_box.find('xmin').text)),
                        int(float(bnd_box.find('ymin').text)),
                        int(float(bnd_box.find('xmax').text)),
                        int(float(bnd_box.find('ymax').text))
                    ]
            labels.append(class_label)
            bboxes.append(bbox)
    return bboxes,labels

def rewritexml(xmlfile,bboxes,labels,patch_w,patch_h,save_abs_name):

    DomTree = ET.parse(xml_root+xmlfile)
    root = DomTree.getroot()
    # new_tree = ET.ElementTree(root)

    ref_object = root.findall('object')[0]
    for objects in root.findall('object'):
        root.remove(objects)
    root.find('size').find('width').text = str(patch_w)
    root.find('size').find('height').text = str(patch_h)
    for bbox,label in zip(bboxes,labels):
        ref_object.find('name').text = label
        bnd_box = ref_object.find('bndbox')
        bnd_box.find('xmin').text = str(bbox[0])
        bnd_box.find('ymin').text = str(bbox[1])
        bnd_box.find('xmax').text = str(bbox[2])
        bnd_box.find('ymax').text = str(bbox[3])
        root.append(ref_object)

    DomTree.write(save_abs_name,encoding='utf-8')

    

for i in tqdm(range(len(files))):
    f = files[i]
    img_i = cv2.imread('/data/wulianjun/StateGridv5/小金具/JPEGImages/'+f.replace('.xml','.jpg'))
    bboxes,labels = readXml(f)
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    img_name = f.replace('.xml','')

    # if len(bboxes)<3:
    #     continue
    
    # img_i_vis = img_i.copy()
    # for box in bboxes:
        # img_i_vis = cv2.rectangle(img_i_vis,(box[0],box[1]),(box[2],box[3]),(0,0,0),thickness=8)
    # from pdb import set_trace
    # set_trace()
    # cv2.imwrite('./1212.jpg',img_i_vis)
    H,W,_ = img_i.shape
    patch_h,patch_w = H//4,W//4
    if len(bboxes) == 0:
        continue
    centers = np.vstack([(bboxes[...,0]+bboxes[...,2])/2,(bboxes[...,1]+bboxes[...,3])/2]).T
    inds = (centers / np.array([patch_w,patch_h])).astype(np.long)

    patches_valid = np.zeros((4,4))
    patches_valid[inds[:,0],inds[:,1]] = 1

    bboxes = bboxes - np.array([patch_w,patch_h,patch_w,patch_h]) * np.hstack([inds,inds])
    patch_ind = np.unique(inds,axis=0)
    for i,idx in enumerate(patch_ind):
        # patches_bboxes.append(bboxes[np.all(inds==idx,axis=1)])
        patches_bbox = bboxes[np.all(inds==idx,axis=1)]
        patches_labels = labels[np.all(inds==idx,axis=1)]

        # bbox_border = np.hstack([idx,idx+1]) * np.array([patch_w,patch_h,patch_w,patch_h])
        patches_bbox[:,::2] = np.clip(patches_bbox[:,::2],0,patch_w)
        patches_bbox[:,1::2] = np.clip(patches_bbox[:,1::2],0,patch_h)
        img_i_patches = img_i[idx[1]*patch_h:(idx[1]+1)*patch_h,idx[0]*patch_w:(idx[0]+1)*patch_w,:]

        # img_vis = img_i_patches.copy()
        # for box in patches_bbox:
        #     img_vis = cv2.rectangle(img_vis,(box[0],box[1]),(box[2],box[3]),(0,0,0),thickness=8)
        # cv2.imwrite('./1212_{}.jpg'.format(i),img_i_vis)
        cv2.imwrite('{}JPEGImages/{}_{}.jpg'.format(target_root,img_name,i),img_i_patches)
        rewritexml(f,patches_bbox,patches_labels,patch_w,patch_h,'{}Annotations/{}_{}.xml'.format(target_root,img_name,i))
        all_xml_file.append('{}_{}'.format(img_name,i))
        
        # from pdb import set_trace
        # set_trace()
    ff = open('/data/wulianjun/StateGridv5/xiaojinju_patch/ImageSets/Main/trainval.txt','w')
    for line in all_xml_file:
        ff.write(line+'\n')
    ff.close()
    # np.where(patches_valid)
