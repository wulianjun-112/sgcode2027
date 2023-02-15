import os.path as osp
import os
import xml.etree.ElementTree as ET
# dataset_names = ['jichu','daodixian','jiedizhuangzhi','ganta','fushusheshi']
dataset_names = ['ganta']

# dataset_train_classes = [
# ['000000151','000000011', '000000061','000000021'],
# ['020001011', '020000031', '020001031', '020100031', '020000011', 
# '020000111', '020000021', '020001061', '020100051', '020100011', 
# '020001021', '020100021'],
# ['050000011'],
# ['010002051', '010000021'],
# ['070002011', '070400021', '070000011', '070000021', '070002021']

# ]

# dataset_val_classes = [['000000151' ,'000000011', '000000061', '000000021'],
# ['020000111' , '020001013' ,'020000031' ,'020001031', 
# '020100032','020000013', '020000023' ,'020000113'],
# ['050000011'],
# ['010002051','010000021','010002052','010000023'],
# ['070002011' ,'070400021','070000011','070000021','070002021']
# ]

dataset_val_classes = [['010000032','010000023']]

dataroot ='/home/public/dataset/stateGridv4/ValSet/'
path = dataroot + 'JPEGImages/'
xml_path_root = dataroot + 'Annotations/'
target_txt_paths =['/home/public/dataset/stateGridv4/ValSet/ImageSets/{}/test.txt'.format(dataset_name) for dataset_name in dataset_names]


def read_xml(xml_path,dataset_val_class):
    DomTree = ET.parse(xml_path)
    root = DomTree.getroot()
    for objects in root.findall('object'):
        label = objects.find('name').text
        if label in dataset_val_class:
            return xml_path

    #     bbox = [
    #         int(bnd_box.find('xmin').text),
    #         int(bnd_box.find('ymin').text),
    #         int(bnd_box.find('xmax').text),
    #         int(bnd_box.find('ymax').text)
    #     ]
    #     bboxes.append(np.array(bbox))
    # bboxes = np.array(bboxes)
    return False

all_files = []

for root, dirs, files in os.walk(xml_path_root, topdown=False):
    for file in files:
        all_files.append(os.path.join(root,file))
from pdb import set_trace
set_trace()
for i,target_txt_path in enumerate(target_txt_paths) :
    if not os.path.exists(os.path.dirname(target_txt_path)):
        os.mkdir(os.path.dirname(target_txt_path))
    
    f = open(target_txt_path,'w')
    for xml_file in all_files:
        save_xml = read_xml(xml_file,dataset_val_classes[i])
        if save_xml:
            f.write(os.path.basename(save_xml).split('.')[0]+'\n')
    f.close()