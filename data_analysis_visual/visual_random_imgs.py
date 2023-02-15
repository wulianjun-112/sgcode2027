#encoding=utf-8
import argparse
import os
from re import I
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
def readXml(xml_path,img_root,count,val_set=False):

    DomTree = ET.parse(xml_path)

    root = DomTree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    # objectlist = annotation.getElementsByTagName('object')
    bboxes = []
    font = ImageFont.truetype('/data/wulianjun/code/simsun.ttf',40,encoding='utf-8')
    count_one =False
    for objects in root.findall('object'):
        difficult = objects.find('difficult').text

        if difficult == '1':
            # if xmlfile not in difficult_obj.keys():
            #     difficult_obj[xmlfile] = 1
            # else:
            #     difficult_obj[xmlfile] += 1
            continue
        else:
            class_label = objects.find('name').text

            # if class_label in classNames:
            # if class_label != args.label:
            #     continue
            # else:
            bnd_box = objects.find('bndbox')
            if class_label == args.label:
                count_one = True

                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text)),
                    class_label,
                    True
                ]
            else:
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text)),
                    class_label,
                    False
                ]
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or min(bbox[:4])<0:
                continue
            else:
                bboxes.append(bbox)

    if count_one:
        count += 1

    if len(bboxes) !=0 and count_one:
        img = cv2.imread(os.path.join(img_root,os.path.basename(xml_path).split('.')[0]+'.jpg'))
        for bbox in bboxes:
            # cv2.putText(img, bbox[-1], (bbox[0], bbox[1]), font, 2, (0,255,0), 3)
            if bbox[-1]:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(255,0,0), 4)
            else:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,0,255), 4)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.text((bbox[0], bbox[1]-20),bbox[-2],(0,0,0),font=font)
        img = np.array(img)
        # from pdb import set_trace
        # set_trace()
        save_dir ='/data/wulianjun/code/CBNetV2/data_analysis_visual/{}/visual_random_imgs/'.format(xml_path.split('/')[-3])
        os.makedirs(save_dir,exist_ok=True)
        # from pdb import set_trace
        # set_trace()
        save_path = save_dir+'{}_'.format(
            args.label)+os.path.basename(xml_path).split('.')[0]+'.jpg' if not val_set else save_dir+'验证集_{}_'.format(
            args.label)+os.path.basename(xml_path).split('.')[0]+'.jpg'

        cv2.imwrite(save_path,img)    
            
    return count
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations file')
        
    parser.add_argument(
        '--label',
        type=str,
        help='which label to count')

    parser.add_argument(
        '--visualization_num',
        default= 1 ,
        type=int,
        help='')

    parser.add_argument(
        '--val_set',
        default= False ,
        action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
count = 0

if args.anno_dir.endswith('txt'):
    f = open(args.anno_dir,'r')
    
    xml_paths = []
    dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))
    img_root = os.path.join(dirname, "JPEGImages")
    for line in f:
        xml_paths.append(os.path.join(dirname, "Annotations", line.split('\n')[0] + ".xml"))

    f.close()
    
    inds = np.arange((len(xml_paths)))
    np.random.shuffle(inds)
    J=0
    while(count<args.visualization_num):
        index = inds[J:1+J]
        vis_xml_paths = [xml_paths[i] for i in index]
        try:
            count = readXml(vis_xml_paths[0],img_root,count)
        except:
            from pdb import set_trace
            set_trace()
        J+=1
        if J>=len(inds):
            break
        
elif os.path.isdir(args.anno_dir):
    all_files = []
    for root, dirs, files in os.walk(args.anno_dir, topdown=False):
        all_files.append(files)
    xml_paths = [os.path.join(args.anno_dir,file) for file in all_files[0]]
    img_root = os.path.join(os.path.dirname(args.anno_dir),'JPEGImages/')
    # from pdb import set_trace
    # set_trace()
    inds = np.arange((len(xml_paths)))
    np.random.shuffle(inds)
    J=0

    while(count<args.visualization_num):
        if J % 20 == 0:
            print(J)
        index = inds[J:1+J]
        vis_xml_paths = [xml_paths[i] for i in index]
        try:
            count = readXml(vis_xml_paths[0],img_root,count,args.val_set)
        except:
            from pdb import set_trace
            set_trace()
        J+=1
        if J>=len(inds):
            break

# print(args.label,':',count)