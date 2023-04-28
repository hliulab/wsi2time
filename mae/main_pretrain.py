# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import openslide
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
import timm
import pandas as pd
#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae

from mae.engine_pretrain import train_one_epoch

def get_args_parser():
    data_slide_dir = "./slides/TCGA-LUNG"
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--data', default="DATA_DIRECTORY", type=str, metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--csv_path', type=str, default='../dataset_csv/sample_data.csv')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--data_h5_dir', type=str, default='../tile_result')
    parser.add_argument('--data_slide_dir', type=str, default=f"{data_slide_dir}")
    parser.add_argument('--model_path', type=str, default="MAE_MODEL")
    parser.add_argument('--data_type', type=str, default="BRCA")
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.2, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--pin_mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
 
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])





    csv_path =args.csv_path
    if csv_path is None:
        raise NotImplementedError("default csv_path is not implemented!")

    print(csv_path)
    bags_dataset = Dataset_All_Bags(csv_path) #返回文件名
    total = len(bags_dataset)


    files = os.listdir(args.data_slide_dir)
    txt_file = None
    for f in files:
        if f.split(".")[-1] == "txt":
            txt_file = f
    assert txt_file is not None
    kidney_info = pd.read_csv(os.path.join(args.data_slide_dir, txt_file), sep="\t")
    slide_to_dir = dict(zip(kidney_info["filename"].values, kidney_info["id"].values))
    # dir_to_slide = dict(zip(kidney_info["id"].values, kidney_info["filename"].values))

    data_loader_train = []
    total_length = 0
    dataset_list=[]
    for bag_candidate_idx in range(total): #一张一张silde读取他的所有batch
        slide_id = bags_dataset[bag_candidate_idx]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', args.data_type, bag_name)
        print(f"读取h5 {h5_file_path}")

        slide_filename = slide_id + ".svs"
        temp_dir = slide_to_dir[slide_filename]
        slide_file_path = os.path.join(args.data_slide_dir, temp_dir, slide_filename)
        wsi = openslide.open_slide(slide_file_path)

        dataset_train = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True,custom_transforms=transform_train)

        kwargs = {'num_workers': 25, 'pin_memory': True} if device.type == "cuda" else {}
        #train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, **kwargs)
        #total_length += (len(dataset_train))
        #data_loader_train.append(train_loader)
        #print(len(data_loader_train))
        dataset_list.append(dataset_train)
    train_dataset = ConcatDataset(dataset_list)
    total_length += (len(train_dataset))
    print(f"一共有{len(dataset_list)}张wsi")
    print(f"一共有{total_length}张patch")

    global_rank=0
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    #if global_rank == 0 and args.log_dir is not None:
        #os.makedirs(args.log_dir, exist_ok=True)
        #log_writer = SummaryWriter(log_dir=args.log_dir)
   # else:
    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
