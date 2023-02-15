# coding=utf-8
import os
import cv2
import numpy as np

import xml.etree.ElementTree as ET

Crop_size = (1333, 800)
keep_radio = True
path = "./JPEGImages_half"
xml_path_root = './Annotations_half'
# target_path = "/home/public/dataset/stateGridv4/tongdao_crop/JPEGImages/"
# target_xml_path = "/home/public/dataset/stateGridv4/tongdao_crop/Annotations/"

from pdb import set_trace
set_trace()


def read_xml(xml_path):
    bboxes = []
    DomTree = ET.parse(xml_path)
    root = DomTree.getroot()

    size = root.find('size')

    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for objects in root.findall('object'):
        bnd_box = objects.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bboxes.append(np.array(bbox))
    bboxes = np.array(bboxes)
    return width, height, bboxes


def resize_xml(xml_path, new_wight, new_height, crop_region):
    DomTree = ET.parse(xml_path)
    root = DomTree.getroot()
    size = root.find('size')

    size.find('width').text = str(int(new_wight))
    size.find('height').text = str(int(new_height))

    for objects in root.findall('object'):
        bnd_box = objects.find('bndbox')
        bnd_box.find('xmin').text = str(
            int(bnd_box.find('xmin').text) - crop_region[0])
        bnd_box.find('ymin').text = str(
            int(bnd_box.find('ymin').text) - crop_region[1])
        bnd_box.find('xmax').text = str(
            int(bnd_box.find('xmax').text) - crop_region[0])
        bnd_box.find('ymax').text = str(
            int(bnd_box.find('ymax').text) - crop_region[1])

    DomTree.write(os.path.join(target_xml_path, os.path.basename(xml_path)))


files = os.listdir(path)
i = 0
for file in files:
    # if i % 10 ==0:
    print(i)
    i += 1

    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
        continue
    else:
        ext = os.path.splitext(file)
        # if ext[0] != '003540':
        #     continue
        # else:
        #     from pdb import set_trace
        #     set_trace()
        if ext[1] != '.jpg' and ext[1] != '.JPG':
            continue
        img = cv2.imread(file_path)
        xml_path = os.path.join(xml_path_root, ext[0] + '.xml')
        # if ext[1] == '.jpg':

        #     new_name = os.path.join(target_path, file)
        #     aa = 0
        # elif ext[1] == '.JPG':
        #     new_name = ext[0] + '.jpg'
        #     new_name = os.path.join(target_path, new_name)

        # cv2.imwrite(new_name,img_resize)

        width, height, bboxes = read_xml(xml_path)
        crop_area = [
            bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(),
            bboxes[:, 3].max()
        ]

        crop_w_h = [crop_area[2] - crop_area[0], crop_area[3] - crop_area[1]]
        Crop_size = (1333, height / width *
                     800) if width / height >= 1 else (1333 * width / height,
                                                       800)
        if crop_w_h[0] < Crop_size[0] and crop_w_h[1] < Crop_size[1]:
            center = [(crop_area[0] + crop_area[2]) / 2,
                      (crop_area[1] + crop_area[3]) / 2]
            if width / height >= 1:  #(Crop_size[0],720)
                crop_area = [
                    center[0] - Crop_size[0] / 2,
                    center[1] - Crop_size[0] * height / width / 2,
                    center[0] + Crop_size[0] / 2,
                    center[1] + Crop_size[0] * height / width / 2
                ]
            else:
                crop_area = [
                    center[0] - Crop_size[1] * width / height / 2,
                    center[1] - Crop_size[1] / 2,
                    center[0] + Crop_size[1] * width / height / 2,
                    center[1] + Crop_size[1] / 2
                ]

            crop_area = np.array(crop_area).astype(np.int32)
            delta = crop_area - np.array([0, 0, width, height])
            sig = np.array([1, 1, -1, -1])
            invalid = (sig * delta) < 0
            if len(np.where(invalid)[0]) != 0:
                invalid_1 = invalid[[2, 3, 0, 1]]
                invalid_2 = invalid | invalid_1

                if len(delta[::2][invalid[::2]]) != 0:
                    crop_area[::2] = crop_area[::2] - invalid_2[::2].astype(
                        np.int32) * delta[::2][invalid[::2]]
                if len(delta[1::2][invalid[1::2]]) != 0:
                    crop_area[1::2] = crop_area[1::2] - invalid_2[1::2].astype(
                        np.int32) * delta[1::2][invalid[1::2]]
                # if len(np.where(invalid)[0]) == 2:
                #     crop_area = crop_area - invalid_2.astype(np.int32) * delta[invalid].repeat(2)[[0,2,1,3]]
                # else:
                #     crop_area = crop_area - invalid_2.astype(np.int32) * delta[invalid]

            # crop_area[::2] = np.clip(crop_area[::2],0,width)
            # crop_area[1::2] = np.clip(crop_area[1::2],0,height)

            crop_area = np.array(crop_area).astype(np.uint32)
        else:
            if len(bboxes) == 1:
                crop_area = [0, 0, width, height]
            else:
                center = np.array([(bboxes[:, 2] + bboxes[:, 0]) / 2,
                                   (bboxes[:, 3] + bboxes[:, 1]) / 2])
                center = center.transpose()

        from pdb import set_trace
        set_trace()
        img_crop = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
        # img_img_crop = np.concatenate((img, img_crop), axis=1)


        cv2.imshow('img', img)
        cv2.moveWindow('img', 0, 0)
        cv2.imshow('img_crop', img_crop)
        cv2.moveWindow('img_crop', 0, 1200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # resize_xml(xml_path,crop_area[2]-crop_area[0],crop_area[3]-crop_area[1],crop_area)
        # img_crop = img[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2],:]

        # cv2.imwrite(new_name,img_crop)
