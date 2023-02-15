import numpy as np
import cv2
import os
from tqdm import tqdm 
import xml.etree.ElementTree as ET
from multiprocessing import Process,current_process
import time
CLASSES  = ['线松股','线异物','线断股','线损伤']
def readXml(xmlfile,xml_root):

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

def rewritexml(xmlfile,xml_root,bboxes,labels,patch_w,patch_h,save_abs_name):

    DomTree = ET.parse(xml_root+xmlfile)
    root = DomTree.getroot()

    # new_tree = ET.ElementTree(root)
    if len(bboxes) >0:
        ref_object = root.findall('object')[0]
        for objects in root.findall('object'):
            root.remove(objects)
        for bbox,label in zip(bboxes,labels):
            ref_object.find('name').text = label
            bnd_box = ref_object.find('bndbox')
            bnd_box.find('xmin').text = str(bbox[0])
            bnd_box.find('ymin').text = str(bbox[1])
            bnd_box.find('xmax').text = str(bbox[2])
            bnd_box.find('ymax').text = str(bbox[3])
            root.append(ref_object)
    else:
        for objects in root.findall('object'):
            root.remove(objects)

    root.find('size').find('width').text = str(patch_w)
    root.find('size').find('height').text = str(patch_h)
    

    DomTree.write(save_abs_name,encoding='utf-8')

    

def tar(files,target_root,xml_root):

    for i in tqdm(range(len(files))):
        f = files[i]
        # img_i = np.random.randn(1000,2000,3)
        # img_i = 1
        img_i = cv2.imread(xml_root.replace('Annotations','JPEGImages')+f.replace('.xml','.jpg'))

        if img_i is None:
            continue
        try:
            bboxes,labels = readXml(f,xml_root)
        except:
            continue

        bboxes = np.array(bboxes)
        labels = np.array(labels)
        img_name = f.replace('.xml','')

        H,W,_ = img_i.shape
        # if len(bboxes) == 0:
        #     continue
        
        if H // 800 <2 :

            cv2.imwrite('{}JPEGImages_crop_no_fliter/{}_{}.jpg'.format(target_root,img_name,0),img_i)
            rewritexml(f,xml_root,bboxes,labels,W,H,'{}Annotations_crop_no_fliter/{}_{}.xml'.format(target_root,img_name,0))

        else:
            cut_num = H // 800
            patch_h,patch_w = H//cut_num,W//cut_num

            if len(bboxes) == 0:
                patch_ind = np.ones((cut_num,cut_num))
                for i in range(cut_num):
                    for j in range(cut_num):
                        img_i_patches = img_i[i*patch_h:(i+1)*patch_h,j*patch_w:(j+1)*patch_w,:]
                        cv2.imwrite('{}JPEGImages_crop_no_fliter/{}_{}.jpg'.format(target_root,img_name,i*cut_num+j),img_i_patches)
                        rewritexml(f,xml_root,bboxes,labels,patch_w,patch_h,'{}Annotations_crop_no_fliter/{}_{}.xml'.format(target_root,img_name,i*cut_num+j))
            else:

                centers = np.vstack([(bboxes[...,0]+bboxes[...,2])/2,(bboxes[...,1]+bboxes[...,3])/2]).T
                inds = (centers / np.array([patch_w,patch_h])).astype(np.long)
                patches_valid = np.zeros((cut_num,cut_num))
                patches_valid[inds[:,0],inds[:,1]] = 1
                bboxes = bboxes - np.array([patch_w,patch_h,patch_w,patch_h]) * np.hstack([inds,inds])

                # patch_ind = np.unique(inds,axis=0)
                patch_ind = np.stack([np.arange(cut_num).repeat(cut_num).reshape(-1,cut_num),np.arange(cut_num).repeat(cut_num).reshape(-1,cut_num).T]).reshape(2,cut_num*cut_num).T
                
                for i,idx in enumerate(patch_ind):
                    # patches_bboxes.append(bboxes[np.all(inds==idx,axis=1)])
                    patches_bbox = bboxes[np.all(inds==idx,axis=1)]
                    patches_labels = labels[np.all(inds==idx,axis=1)]

                    # bbox_border = np.hstack([idx,idx+1]) * np.array([patch_w,patch_h,patch_w,patch_h])
                    patches_bbox[:,::2] = np.clip(patches_bbox[:,::2],0,patch_w)
                    patches_bbox[:,1::2] = np.clip(patches_bbox[:,1::2],0,patch_h)
                    img_i_patches = img_i[idx[1]*patch_h:(idx[1]+1)*patch_h,idx[0]*patch_w:(idx[0]+1)*patch_w,:]

                
                    cv2.imwrite('{}JPEGImages_crop_no_fliter/{}_{}.jpg'.format(target_root,img_name,i),img_i_patches)
     
                    rewritexml(f,xml_root,patches_bbox,patches_labels,patch_w,patch_h,'{}Annotations_crop_no_fliter/{}_{}.xml'.format(target_root,img_name,i))



def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]

if __name__ == '__main__':

    xml_root = '/data/wulianjun/datasets/SG5/导地线/Annotations/'
    files_all = os.listdir(xml_root)
    
    
    target_root = '/data/wulianjun/datasets/SG5/导地线/'
    os.makedirs(target_root+'JPEGImages_crop_no_fliter',exist_ok=True)
    os.makedirs(target_root+'Annotations_crop_no_fliter',exist_ok=True)

    mp_num = 32
    files =  split_list_average_n(files_all, n=len(files_all)//mp_num)

    files_patches = [[] for _ in range(mp_num+1)]
    for i,file in enumerate(files):
        files_patches[i].extend(file)

    files_patches[-2].extend(files_patches[-1])
    files_patches = files_patches[:-1]
    # tar(files_patches[0],target_root,xml_root)
    p_list = []
    for i in range(mp_num):
        p = Process(target=tar,args=(files_patches[i],target_root,xml_root))
        p.start()
        p_list.append(p)
    # ff = open('/data/wulianjun/StateGridv5/xiaojinju_patch/ImageSets/Main/trainval.txt','w')
    
    # for line in all_xml_file:
    #     ff.write(line+'\n')
    # ff.close()
    for p in p_list:
        p.join()
    print('main done!')

    from pdb import set_trace
    set_trace()
    def Split(str):
        re = ''
        for x in str.split('_')[:-1]:
            re += x
        return re
    all_xml = os.listdir(target_root+'Annotations_crop_no_fliter')
    with open(target_root+'ImageSets/Main/train_split.txt','r') as ff:
        ab = ff.readlines()

    with open(target_root+'ImageSets/Main/train_crop_no_fliter.txt','w') as f:
        for xml in all_xml:
            if Split(xml)+'\n' in ab:
                f.write(xml.split('.xml')[0]+'\n')
    
    with open(target_root+'ImageSets/Main/val_split.txt','r') as ff:
        ab = ff.readlines()
    with open(target_root+'ImageSets/Main/val_crop_no_fliter.txt','w') as f:
        for xml in all_xml:
            if Split(xml)+'\n' in ab:
                f.write(xml.split('.xml')[0]+'\n')