import argparse
import os
import os.path as osp
import time
import warnings
from loguru import logger
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import csv
import sys
import json

'''
CUDA_VISIBLE_DEVICES=2 python tools/test.py --model_types 03 --input_pic_path /data/taoranyi/temp_for_wulianjun/datasets/SG5/导地线/JPEGImages --input_pic_json /data/taoranyi/temp_for_wulianjun/datasets/SG5/导地线/pictureName.json --output_dir /data/taoranyi/temp_for_wulianjun/sgcode2027/output_dir/ --eval mAP
'''

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')

    parser.add_argument('--model_types',type=str,default='00')
    parser.add_argument('--allowed_model_types',type=str,default='00,01,02,03,04,05,06,07')
    parser.add_argument('--input_pic_path',type=str,default='/usr/input_picture')
    parser.add_argument('--input_pic_json',type=str,default='/usr/input_picture_attach/pictureName.json')
    parser.add_argument('--output_dir',type=str,default='/usr/output_dir/')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out',action='store_true',help='output result file in csv file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

@logger.catch
def main(args):
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        # raise ValueError('The output file must be a pkl file.')

    def get_config_cpt_path(args):
        config_paths = []
        cpt_paths = []
        model_types = args.model_types.split(',')
        allowed_model_types = args.allowed_model_types.split(',')
        allowed_name = {'00':'jichu','01':'ganta','02':'daodixian','03':'jueyuanzi','04':'jinju','05':'jiedizhuangzhi','06':'tongdaohuanjing','07':'fushusheshi'}
        for x in model_types:
            if x not in allowed_model_types:
                raise ValueError('Not support for {} type'.format(x))
        
            config_paths.append('configs/SG5/{}/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_{}.py'.format(allowed_name[x],allowed_name[x]))
            cpt_paths.append('work_dirs/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_{}/epoch_9.pth'.format(allowed_name[x]))
        return config_paths, cpt_paths

    config_paths, cpt_paths = get_config_cpt_path(args)

    #TODO support for muilt-type test
    if not len(config_paths) == len(cpt_paths) == 1:
        raise NotImplementedError

    cfg = Config.fromfile(config_paths[0])

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')


    assert os.path.exists(args.input_pic_json) , 'Can not find pictureName.json !'
    with open(args.input_pic_json,'r') as f:
        pictureName = json.load(f)

    ori_fileName = []
    for x in pictureName:
        ori_fileName.append(x["ori_fileName"])


    cfg.data.test['img_prefix'] = args.input_pic_path
    cfg.data.test['ann_file'] = args.input_pic_json
    

    # build the dataloader

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    #TODO
    # checkpoint = load_checkpoint(model, cpt_paths[0], map_location='cpu')
    checkpoint = load_checkpoint(model, '/data/taoranyi/temp_for_wulianjun/sgcode2027/docker/work_dirs/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian/epoch_9.pth', map_location='cpu')
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)




    rank, _ = get_dist_info()
    if rank == 0:

        if args.out:
            # print(f'\nwriting results to {args.out}')
            # mmcv.dump(outputs, args.out)
            with open(osp.join(args.output_dir,'result.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerow(['filename','name','score','xmin','ymin','xmax','ymax'])
                for img_i in outputs:
                    file_name = img_i[-1]
                    img_i = img_i[:-1]
                    for cls_i,name in zip(img_i,model.module.CLASSES):
                        aa = cls_i[cls_i[:,-1] > args.show_score_thr]
                        aa = aa[:,[4,0,1,2,3]]
                        prit = [(file_name,name,*x) for x in aa]
                        writer.writerows(prit)

        for i in range(len(outputs)):
            outputs[i] = outputs[i][:-1]
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

if __name__ == '__main__':

    def setup_logger(output=None,logger_name="program_log.txt"):
        """
        Initialize the cvpods logger and set its verbosity level to "INFO".

        Args:
            output (str): a file name or a directory to save log. If None, will not save log file.
                If ends with ".txt" or ".log", assumed to be a file name.
                Otherwise, logs will be saved to `output/log.txt`.

        Returns:
            logging.Logger: a logger
        """
        logger.remove()
        loguru_format1 = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{message}</level>"
        )
    #     loguru_format2 = (
    #     "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    #     "<level>{level: <8}</level> | "
    #     "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    # )
        # stdout logging: master only
        logger.add(sys.stderr, format=loguru_format1)
        # logger.add(sys.stderr, format=loguru_format1, filter=lambda x: "·" in x['message'])
        # file logging: all workers
        if output is not None:
            assert logger_name.endswith('.txt')
            # assert result_logger.endswith('.txt')

            open(os.path.join(output, logger_name), 'w').close()

            #TODO
            logger.add(os.path.join(output, logger_name))
            # logger.add(os.path.join(output, result_logger), filter=lambda x: "·" in x['message'])

    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)

    setup_logger(output=args.output_dir)
    main(args)

