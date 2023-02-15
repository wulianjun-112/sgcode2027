import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
xml_root = '/data/wulianjun/StateGridv5/杆塔/Annotations/'

files = os.listdir(xml_root)
CLASSES = ['角杆塔塔材锈蚀', '角杆塔塔身锈蚀', '砼杆叉梁抱箍锈蚀',
'角杆塔脚钉锈蚀','角杆塔塔材变形']
without_yiwu_xmls = []
def readXml(xmlfile):
    global without_yiwu_xmls
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
    if len(bboxes)>0:
        without_yiwu_xmls.append(xmlfile)
    return bboxes,labels


ff = open('./train_wo_yiwu.txt','w')
for i in tqdm(range(len(files))):
    file = files[i]
    readXml(file)
# from pdb import set_trace
# set_trace()
for x in without_yiwu_xmls:
    ff.write(x.replace('.xml','\n'))

ff.close()