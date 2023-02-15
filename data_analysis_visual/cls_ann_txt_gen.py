# coding=utf-8

### based on xxx_crop2 datasets generate an annotation txt for classification

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # python cls_ann_txt_gen.py --anno-dir --class_list '070002011' '070400021' '070000011' '070000021' '070002021'
    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of xxx_crop2 annotations file')
        
    parser.add_argument(
        '--class_list',
        type=str,
        nargs='+',
        help='which labels need to split')
    
    parser.add_argument(
        '--output_txt_name',
        default='trainval_cls.txt',
        type=str,
        help='...')

    args = parser.parse_args()
    return args


def read_rewrite_Xml(xmlfile,save_dir,all_xml_name):

    DomTree = ET.parse(xmlfile)
    root = DomTree.getroot()
    # filename = root.find('filename').text
    size = root.find('size')
    imgwidth = int(size.find('width').text)
    imgheight = int(size.find('height').text)
    # objectlist = annotation.getElementsByTagName('object')
    labels=[]
    w_hs = []
    bboxes = []
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
            if class_label not in args.class_list:
                continue
            else:
                bnd_box = objects.find('bndbox')
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or min(bbox)<0:
                    continue
                else:
                     labels.append(class_label)
                     w_hs.append((bbox[2]-bbox[0],bbox[3]-bbox[1]))
                     bboxes.append(bbox)

    if len(root.findall('object'))>0:
        ref_object = root.findall('object')[0]              
    for objects in root.findall('object'):
        root.remove(objects)
    

    



    if len(labels)>=1:
        for index,(w_h,label,bbox) in enumerate(zip(w_hs,labels,bboxes)):
            W,H = w_h
            
            W_ = H if H/W >=2 else W
            H_ = W if W/H >=2 else H
            (W_,H_) = (min((imgwidth,W_)),min((imgheight,H_)))
            if max(W_,H_) <= 168:
                (W_,H_) = (int(W_ / H_ * 168), 168) if W_ >= H_ else (168, int(H_ / W_) * 168)
            center = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
            crop_area = (center[0]-W_//2,center[1]-H_//2,center[0]+W_//2,center[1]+H_//2)
            crop_area = np.floor(np.array(crop_area)).astype(np.int32)
            delta = crop_area - np.array([0, 0, imgwidth, imgheight])
            sig = np.array([1, 1, -1, -1])
            invalid = (sig * delta) < 0
            if len(np.where(invalid)[0]) != 0:
                invalid_1 = invalid[[2, 3, 0, 1]]
                invalid_2 = invalid | invalid_1
                if len(delta[::2][invalid[::2]]) != 0:
                    crop_area[::
                                2] = crop_area[::2] - invalid_2[::2].astype(
                                    np.int32) * delta[::2][invalid[::2]]
                if len(delta[1::2][invalid[1::2]]) != 0:
                    crop_area[1::2] = crop_area[1::2] - invalid_2[
                        1::2].astype(np.int32) * delta[1::2][invalid[1::2]]
            if (crop_area<0).any():
                from pdb import set_trace
                set_trace()
            bbox = crop_area.tolist()
            ref_object.find('name').text = label
            bnd_box = ref_object.find('bndbox')
            bnd_box.find('xmin').text = str(bbox[0])
            bnd_box.find('ymin').text = str(bbox[1])
            bnd_box.find('xmax').text = str(bbox[2])
            bnd_box.find('ymax').text = str(bbox[3])
            # name = ET.Element('object')
            # name.text = label
            root.append(ref_object)
            root.find('size').find('width').text = str(w_h[0])
            root.find('size').find('height').text = str(w_h[1])
            xml_name = '{}_{}'.format(os.path.basename(xml_path).split('.')[0],index)
            write_dir = os.path.join(save_dir, xml_name+'.xml')
            DomTree.write(write_dir,encoding='utf-8')
            all_xml_name.append(xml_name)

    return all_xml_name



args = parse_args()

assert args.anno_dir.endswith('txt')
xml_save_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))
xml_save_dir = os.path.join(xml_save_dir,'Annotations_cls')
if not os.path.exists(xml_save_dir):
    os.mkdir(xml_save_dir)

txt_save_dir = os.path.dirname(args.anno_dir)
txt_path = os.path.join(txt_save_dir,args.output_txt_name)
assert not os.path.exists(txt_path),'annotation txt cant override!!!'
 
f = open(args.anno_dir,'r')

xml_paths = []
dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))

for line in f:
    xml_paths.append(os.path.join(dirname, "Annotations", line.split('\n')[0] + ".xml"))


f.close()
from pdb import set_trace
set_trace()
all_xml_name = []
for xml_path in xml_paths:
    all_xml_name = read_rewrite_Xml(xml_path,xml_save_dir,all_xml_name)

from pdb import set_trace
set_trace()
f = open(txt_path,'w')

for xml_name in all_xml_name:
    f.write(xml_name+'\n')

f.close()