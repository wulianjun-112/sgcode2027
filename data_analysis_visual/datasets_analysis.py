import argparse
import json
import os
import os.path as osp
import numpy as np
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import megfile
# python datasets_analysis.py --anno-dir /home/lianjun.wu/wulianjun/mmdetection/voc0712/pascal_trainval0712.json  --test-anno-dir /home/lianjun.wu/wulianjun/VOCdevkit/VOC2007/Annotations/VOC07_test.json

train_classes = ['线异物'
,'线断股'
,'线松股'
,'线损伤']

# val_classes = ['020000111' , '020001013' ,'020000031' ,'020001031', 
# '020100032','020000013', '020000023' ,'020000113']
val_classes = train_classes

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations file')
        
    
    parser.add_argument(
        '--test-anno-dir',
        type=str,
        help='path of test annotations file')

    parser.add_argument(
        '--dirname',
        type=str,
        help='path of jpg and xml for voc type datasets')

    parser.add_argument(
        '--output-name',
        type=str,
        help='path of jpg and xml for voc type datasets')

    args = parser.parse_args()
    return args


def Q_Q_plot(dataset1_labels,dataset2_labels):
# df_samp, df_clu are two dataframes with input data set
    # ref = np.concatenate(dataset1_labels)
    # samp = np.concatenate(dataset2_labels)
    ref = dataset1_labels
    samp = dataset2_labels
    ref_id = ref.shape[0]
    samp_id = samp.shape[0]
    # theoretical quantiles
    samp_pct_x = np.asarray([percentileofscore(ref, x) for x in samp])
    # sample quantiles
    samp_pct_y = np.asarray([percentileofscore(samp, x) for x in samp])
    # estimated linear regression model
    p = np.polyfit(samp_pct_x, samp_pct_y, 1)
    regr = LinearRegression()
    model_x = samp_pct_x.reshape(len(samp_pct_x), 1)
    model_y = samp_pct_y.reshape(len(samp_pct_y), 1)
    regr.fit(model_x, model_y)
    r2 = regr.score(model_x, model_y)
    # get fit regression line
    if p[1] > 0:
        p_function = "y= %s x + %s, r-square = %s" %(str(p[0]), str(p[1]), str(r2))
    elif p[1] < 0:
        p_function = "y= %s x - %s, r-square = %s" %(str(p[0]), str(-p[1]), str(r2))
    else:
        p_function = "y= %s x, r-square = %s" %(str(p[0]), str(r2))
    print("The fitted linear regression model in Q-Q plot using data from enterprises {} and cluster {} is {}".format(str(samp_id), str(ref_id), p_function))

    # plot q-q plot
    x_ticks = np.arange(0, 100, 20)
    y_ticks = np.arange(0, 100, 20)
    plt.scatter(x=samp_pct_x, y=samp_pct_y, color='blue')
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    # add fit regression line
    plt.plot(samp_pct_x, regr.predict(model_x), color='red', linewidth=2)
    # add 45-degree reference line
    plt.plot([0, 100], [0, 100], linewidth=2)
    plt.text(10, 70, p_function)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel('cluster quantiles - id: {}'.format(str(ref_id)))
    plt.ylabel('sample quantiles - id: {}'.format(str(samp_id)))
    plt.title('{} VS {} Q-Q plot'.format(str(ref_id), str(samp_id)))
    plt.savefig('./{}.jpg'.format(args.output_name))


def hist_plot(labels,output_name=None):

    # labels = np.concatenate(labels)
    labels.sort()
    plt.figure()
    plt.hist(labels, bins=max(labels)+1)

    plt.xlabel('label_id')
    plt.ylabel('count')
    if output_name:
        plt.savefig('./{}.jpg'.format(output_name))
    else:
        plt.savefig('./{}.jpg'.format(args.output_name))


args = parse_args()
train_cat2label =  {cat: i for i, cat in enumerate(train_classes)}
val_cat2label =  {cat: i for i, cat in enumerate(val_classes)}
assert args.anno_dir is not None
train_data_infos = []
train_all_bboxes = []
train_all_whs = []
train_all_img_whs = []
train_all_labels = []
resize_keep_ratio = (1333,800)
assert osp.isfile(args.anno_dir) 
if  args.anno_dir.endswith('json'):
    dataset = COCO(args.anno_dir)
    img_ids = dataset.getImgIds()
    cats = dataset.loadCats(dataset.getCatIds())
    label_ids={cat['id']: i for i, cat in enumerate(cats)}

    for i in img_ids:

        info = dataset.loadImgs([i])[0]
        train_all_img_whs.append((info['width'],info['height']))
        img_w,img_h = info['width'],info['height']
        if resize_keep_ratio[0] / img_w * img_h <= 800:
            scale_ratio = resize_keep_ratio[0] / img_w
        else:
            scale_ratio = resize_keep_ratio[1] / img_h

        train_data_infos.append(info)

        ann_ids = dataset.getAnnIds(imgIds=[i])
        bboxes = np.array([ann['bbox'] for ann in dataset.loadAnns(ann_ids)]) 
        labels = np.array([label_ids[ann['category_id']] for ann in  dataset.loadAnns(ann_ids)])
        
        for bbox in bboxes:
            W,H = bbox[2] , bbox[3]
            if W>0 and H>0 :
                train_all_whs.append((W*scale_ratio,H*scale_ratio))
        train_all_bboxes.append(bboxes)
        train_all_labels.append(labels)
elif args.anno_dir.endswith('txt'):
    with open(args.anno_dir,'r',encoding='UTF-8') as f:
        fileids = f.readlines()

        dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid.split('\n')[0] + ".xml")
            tree = ET.parse(anno_file)
            img_w,img_h = int(tree.getroot().find('size').find('width').text),int(tree.getroot().find('size').find('height').text)
            if resize_keep_ratio[0] / img_w * img_h <= 800:
                scale_ratio = resize_keep_ratio[0] / img_w
            else:
                scale_ratio = resize_keep_ratio[1] / img_h
            once = True
            for obj in tree.findall("object"):
                
                cls = obj.find("name").text
                if cls not in train_classes:
                    continue
                if once:
                    train_all_img_whs.append((img_w,img_h))
                    once=False
                train_all_labels.append(train_cat2label[cls])
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                W,H = bbox[2] - bbox[0] , bbox[3] - bbox[1]
                
                if W>0 and H>0 :
                    train_all_whs.append((W*scale_ratio,H*scale_ratio))
                
                train_all_bboxes.append(bbox)
else:
    raise NotImplementedError


if args.test_anno_dir is not None:
    test_data_infos = []
    test_all_bboxes = []
    test_all_labels = []
    assert osp.isfile(args.test_anno_dir) 
    if args.test_anno_dir.endswith('json'):
        dataset = COCO(args.test_anno_dir)
        img_ids = dataset.getImgIds()
        cats = dataset.loadCats(dataset.getCatIds())
        label_ids={cat['id']: i for i, cat in enumerate(cats)}
        for i in img_ids:
            info = dataset.loadImgs([i])[0]
            test_data_infos.append(info)

            ann_ids = dataset.getAnnIds(imgIds=[i])
            bboxes = np.array([ann['bbox'] for ann in dataset.loadAnns(ann_ids)]) 
            labels = np.array([label_ids[ann['category_id']] for ann in  dataset.loadAnns(ann_ids)])
            test_all_bboxes.append(bboxes)
            test_all_labels.append(labels)
    elif args.test_anno_dir.endswith('txt'):
        with open(args.anno_dir,'r',encoding='UTF-8') as f:
            fileids = f.readlines()

            dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.test_anno_dir)))
        for fileid in fileids:
            bboxes = []
            labels = []
            anno_file = os.path.join(dirname, "Annotations", fileid.split('\n')[0] + ".xml")
            tree = ET.parse(anno_file)
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if cls not in val_classes:
                    continue
                test_all_labels.append(val_cat2label[cls] % len(train_classes))
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                bboxes.append(bbox)
    else:
        raise NotImplementedError


def compute_iou(x, centers):
    """ 计算所有x与对应centers数据的交并比
    :param x:
    :param centers:
    :return:
    """
    # 1. x,centers 复制
    box1 = np.reshape(np.tile(np.expand_dims(x, axis=1), [1, 1, np.shape(centers)[0]]), [-1, 2])
    xw, xh = box1[:, 0], box1[:, 1]
    box2 = np.tile(centers, [np.shape(x)[0], 1])
    cw, ch = box2[:, 0], box2[:, 1]
    # 2. 计算交叉面积
    min_w = np.minimum(xw, cw)
    min_h = np.minimum(xh, ch)
    intersection = min_w * min_h
    # 3. 计算交并比
    x_area = xw * xh
    c_area = cw * ch
    union = x_area + c_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    iou = np.reshape(iou, [-1, np.shape(centers)[0]])
    return iou

def kmeans(x, K, save_cluster_fig=""):
    """ 根据一系列宽高数据聚类生成对应K个中心anchor
    :param x:
    :param K:
    :param save_cluster_fig: 输出聚类效果图保存路径, 为空不输出
    :return: [(w, h)¹, (w, h)², ... (w, h)ᵏ]
    """

    # 随机选K个中心点作为初始值
    centers = x[np.random.choice(a=x.shape[0], size=K, replace=False)]
    pre_max_center_ids = np.zeros(np.shape(x)[0], dtype=np.int64)

    step = 0
    while True:
        iou = compute_iou(x=x, centers=centers)
        max_center_ids = np.argmax(a=iou, axis=1)
        print("step {}, centers: \n {}".format(step, centers))

        if np.all(max_center_ids == pre_max_center_ids):
            if save_cluster_fig:
                for i in range(K):
                    plt.scatter(x[max_center_ids == i, 0], x[max_center_ids == i, 1], label=i, s=10)
                plt.scatter(centers[:, 0], centers[:, 1], s=30, color='k')
                plt.legend()
                plt.savefig(save_cluster_fig)
                plt.show()

            # 根据面积大小重新排序输出centers
            sort_centers = sorted(centers, key=lambda v:v[0]*v[1], reverse=False)
            sort_centers = np.array(sort_centers, dtype=np.int64)
            print("final centers \n {}".format(sort_centers))
            return sort_centers

        centers = np.zeros_like(centers, dtype=np.float32)
        for j in range(K):
            target_x_index = max_center_ids == j
            target_x = x[target_x_index,]
            # print(np.sum(target_x, axis=0), np.sum(target_x_index))
            centers[j] = np.sum(target_x, axis=0) / np.sum(target_x_index)

        pre_max_center_ids = max_center_ids.copy()
        step += 1

# train_all_labels = np.concatenate(train_all_labels)
# test_all_labels = np.concatenate(test_all_labels)
# train_all_labels = np.array(train_all_labels)
# test_all_labels = np.array(test_all_labels)
from pdb import set_trace
set_trace()
train_all_whs=np.array(train_all_whs,dtype=np.int32)
train_all_img_whs=np.array(train_all_img_whs,dtype=np.int32)
sort_centers = kmeans(train_all_whs,9)
sort_centers[:,0]/sort_centers[:,1]
sort_centers.mean(1)
kmeans(train_all_img_whs,1)


# train_label_count = [len(np.where(train_all_labels==i)[0]) for i in range(max(train_all_labels)+1)]
# test_label_count = [len(np.where(test_all_labels==i)[0]) for i in range(max(test_all_labels)+1)]
# hist_plot(train_all_labels,'daodixian_train')
# hist_plot(test_all_labels,'daodixian_val')

Q_Q_plot(train_all_labels,test_all_labels)
