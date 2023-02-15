# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import os
import os.path as osp
from re import S
from mmcv.utils import print_log
import xml.etree.ElementTree as ET
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image

def get_gtbbox_scale(gtbbox):
    return (gtbbox[:,3]-gtbbox[:,1])*(gtbbox[:,2]-gtbbox[:,0])


@DATASETS.register_module()
class SG4(XMLDataset):

    def __init__(self, val_classes,merge_label=None,**kwargs):
        super(SG4, self).__init__(**kwargs)
        self.year = 2021

        assert val_classes is not None
        self.val_classes = val_classes
        assert self.CLASSES is not None
        
        self.train_cat2label =  {cat: i for i, cat in enumerate(self.CLASSES)}
        self.val_cat2label =  {cat: i for i, cat in enumerate(self.val_classes)}

        if merge_label is not None:
            self.train_cat2label.update(merge_label)


    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.2,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.val_classes
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            # from pdb import set_trace
            # set_trace()
            # img_urls = [self.img_prefix+x['filename'] for x in self.data_infos]
            # dict_prop = dict()
            # for i in range(len(img_urls)):
            #     dict_prop[img_urls[i]]=results[i]
            # import torch
            # torch.save(dict_prop,'fcos_out.pth')
            gt_bboxes = [ann['bboxes'] for ann in annotations]

            if self.eval_baseon_size:
            
                gt_bboxes_scales = [get_gtbbox_scale(gt_bboxe) for gt_bboxe in gt_bboxes]
 
                gt_bboxes_group = [[],[],[]]
                for img_id,gt_scale in enumerate(gt_bboxes_scales):
                    gt_bboxes_group[0].append(gt_bboxes[img_id][np.where(gt_scale<32*32)[0],:])
                    gt_bboxes_group[1].append(gt_bboxes[img_id][np.where((gt_scale<128*128)&(gt_scale>32*32))[0],:])
                    gt_bboxes_group[2].append(gt_bboxes[img_id][np.where(gt_scale>128*128)[0],:])
                recalls=[]
                proposal_nums=(10,100,300)
                iou_thrs=(0.5,0.7,0.9)

                # repeat_results = []
                # for i in range(len(iou_thrs)):
                #     repeat_results += results

                # array_iou_thrs = np.array(iou_thrs)
                # array_iou_thrs = array_iou_thrs.repeat(len(results))
                # array_iou_thrs = array_iou_thrs.tolist()
                # results_GT_iou_thrs_num = [len(np.where(result[:,4]>iou_thr)[0]) for result,iou_thr in zip(repeat_results,array_iou_thrs)]
                # aa = tuple(len(results) for _ in range(len(iou_thrs)))
                # import torch
                # results_GT_iou_thrs_num = torch.tensor(results_GT_iou_thrs_num)
                # results_GT_iou_thrs_num = results_GT_iou_thrs_num.split(aa,0)

                # results_tensor = torch.tensor(results)
                # for i in range(len(results)):
                #     results[i] = results[i][0]
                for i in range(3):
                    recalls.append(eval_recalls(gt_bboxes_group[i], results, proposal_nums, iou_thrs, logger=logger))

                items = ['S','M','L']
                for k,item in enumerate(items):
                    for i, num in enumerate(proposal_nums):
                        for j, iou_thr in enumerate(iou_thrs):
                            eval_results[f'recall@{item}@{num}@{iou_thr}'] = recalls[k][i, j]
                for k in range(3):
                    if recalls[k].shape[1] > 1:
                        ar = recalls[k].mean(axis=1)
                        for i, num in enumerate(proposal_nums):
                            eval_results[f'AR@{num}'] = ar[i]
            else:

                recalls = eval_recalls(
                    gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
                for i, num in enumerate(proposal_nums):
                    for j, iou_thr in enumerate(iou_thrs):
                        eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
                if recalls.shape[1] > 1:
                    ar = recalls.mean(axis=1)
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
            # gt_bboxes = [ann['bboxes'] for ann in annotations]
            # recalls = eval_recalls(
            #     gt_bboxes,
            #     results,
            #     proposal_nums,
            #     iou_thrs,
            #     logger=logger,
            #     use_legacy_coordinate=True)
            # for i, num in enumerate(proposal_nums):
            #     for j, iou_thr in enumerate(iou_thrs):
            #         eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            # if recalls.shape[1] > 1:
            #     ar = recalls.mean(axis=1)
            #     for i, num in enumerate(proposal_nums):
            #         eval_results[f'AR@{num}'] = ar[i]
        return eval_results


    def get_ann_info(self, idx):
            """Get annotation from XML file by index.

            Args:
                idx (int): Index of data.

            Returns:
                dict: Annotation info of specified index.
            """

            img_id = self.data_infos[idx]['id']

            xml_path = os.path.join(self.img_prefix, self.sub_ann,
                                    f'{img_id}.xml')

            tree = ET.parse(xml_path)
            root = tree.getroot()
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for obj in root.findall('object'):
                name = obj.find('name').text

                if not self.test_mode:
                    if name not in self.train_cat2label.keys():
                        continue
                    label = self.train_cat2label[name]
                else:
                    if name not in self.val_classes:
                        continue
                    label = self.val_cat2label[name] % len(self.CLASSES)
                # if label >= 3 :
                #     label = label - 3
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
                bnd_box = obj.find('bndbox')
                    # TODO: check whether it is necessary to use int
                    # Coordinates may be float type
                bbox = [
                        int(float(bnd_box.find('xmin').text)),
                        int(float(bnd_box.find('ymin').text)),
                        int(float(bnd_box.find('xmax').text)),
                        int(float(bnd_box.find('ymax').text))
                    ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0, ))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
                labels_ignore = np.array(labels_ignore)

            ann = dict(
                        bboxes=bboxes.astype(np.float32),
                        labels=labels.astype(np.int64),
                        bboxes_ignore=bboxes_ignore.astype(np.float32),
                        labels_ignore=labels_ignore.astype(np.int64))

            return ann

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        par = tqdm(img_ids)
        for img_id in par:
            
            if not osp.exists(osp.join(self.img_prefix, self.sub_jpg,'{}.jpg'.format(img_id))):
                filename = f'{self.extra_jpg}/{img_id}.jpg'
            else:
                filename = f'{self.sub_jpg}/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, self.sub_ann,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)

            root = tree.getroot()

            if len(root.findall('object')) ==0:
                group_id = 1
            else:
                group_id = 0

            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, self.sub_jpg,
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height,group_id=group_id))

        return data_infos


    def get_cat2imgs(self,seed=0):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(['pos','neg']))}
        
        for i in range(len(self)):
            group_id = self.data_infos[i]['group_id']
            if group_id==0:
                cat2imgs[0].append(i)
            else:
                cat2imgs[1].append(i)

        negtive_num = len(cat2imgs[0])
        negtive = np.array(cat2imgs[1])
        np.random.seed(seed)
        np.random.shuffle(negtive)
        cat2imgs[1] = negtive[:negtive_num].tolist()
        return cat2imgs