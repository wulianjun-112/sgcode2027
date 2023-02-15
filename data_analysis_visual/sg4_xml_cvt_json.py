import os.path as osp
import mmcv
import xml.etree.ElementTree as ET
import numpy as np
import argparse




train_class = ['色标牌退色', '警告牌图文不清', '防鸟刺未打开', '驱鸟器损坏']
val_classes = train_class


label_ids = {name: i for i, name in enumerate(train_class)}
val_label_ids = {name: i for i, name in enumerate(val_classes)}
# label_ids = [label_ids,val_label_ids]
def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in val_classes:
            continue
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation

def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(train_class):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco

# from pdb import set_trace
# set_trace()

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations txt file')
        

    parser.add_argument(
        '--output-dir',
        type=str,
        help='path of output json files')

    parser.add_argument(
        '--output-name',
        type=str,
        help='name of output json files')

    args = parser.parse_args()
    return args



args = parse_args()
image_roots = [osp.dirname(args.anno_dir)+"/JPEGImages"]

xml_roots = [osp.dirname(args.anno_dir)+"/Annotations"]

filelists = [osp.dirname(args.anno_dir)+"/ImageSets/Main/val.txt"]
out_files = [osp.join(osp.dirname(i),"{}.json".format(args.output_name)) for i in image_roots]

from pdb import set_trace
set_trace()



for SG4_image_root,SG4_xml_root,filelist,out_file in zip(image_roots,xml_roots,filelists,out_files):

    img_names = mmcv.list_from_file(filelist)
    xml_paths = [
        osp.join(SG4_xml_root, f'{img_name}.xml')
        for img_name in img_names
    ]
    img_paths = [
        osp.join(SG4_image_root, f'{img_name}.jpg')
        for img_name in img_names
    ]
    annotations = mmcv.track_progress(parse_xml,list(zip(xml_paths, img_paths)))
    annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, "./{}.json".format(args.output_name))


aa = 0