import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import sys
import math
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import timm
import wandb
from typing import Iterable
import numpy as np

# own-defined library
import utils.misc as misc
from models.model_wrapper import ModelWrapper
from utils.util import AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
from utils.util import set_model, set_loader, set_loader_memory


assert timm.__version__ == "0.3.2"  # version check

best_loss = 1000


def get_args_parser():
    parser = argparse.ArgumentParser('LaCViT contrastive training', add_help=False)

    # Optimizer parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_decay_epochs', type=str, default='4,6,10,12,14',
                        help='where to decay lr, can be a list')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100','imagenet', 'path', 'bird','flowers102','oxfordpet','caltech256'], help='dataset')
    # parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    # parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # MAE params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # distributed training parameters
    parser.add_argument('--distributed', default=False, type=bool,
                        help='whether distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_cluster', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # args for image encoder
    parser.add_argument('--model', type=str, default='mae_cl_base')
    parser.add_argument('--vit_model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mae_chkpt_dir', default='./checkpoints/mae_finetuned_vit_base.pth', type=str,
                        help='checkpoint path for mae model')
    parser.add_argument('--data2vec_chkpt', default='./checkpoints/data2vec_base.pth', type=str,
                        help='checkpoint path for data2vec model')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--simmim_chkpt', default=None, type=str,
                        help='checkpoint path for simmim model')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--method', type=str, default='LaCViT',
                        choices=['LaCViT', 'SimCLR', 'Npair'], help='choose method')

    # other setting
    parser.add_argument('--imagenet_folder', type=str, default='./datasets/imagenet/imagenet/data/CLS-LOC/train',
                        help='path to imagenet dataset')
    parser.add_argument('--train_length', type=float, default=1,
                        help='length of training set')
    parser.add_argument('--project', type=str, default='Test',
                         help='project name for wandb')
    parser.add_argument('--note', type=str, default=' ',
                         help='note for wandb')
    parser.add_argument('--wandb_enable', action='store_true',
                        help='enable wandb to record')
    parser.add_argument('--tag', type=str, default=None,
                         help='Tag for wandb')
    parser.add_argument('--load_data_memory', action='store_true',
                        help='load pre-processed images into memory')
    parser.add_argument('--exchange_path', type=str, default='./exchange/',
                        help='exchange path to save model')
    parser.add_argument('--l2_reg', type=float, default=0.02,
                        help='NPairLoss')
    parser.add_argument('--n_cls', default=None, type=int,
                        help='Number of classes')
    parser.add_argument('--batch_n_cls', default=None, type=int,
                        help='Number of classes per batch')
    parser.add_argument('--batch_n', default=None, type=int,
                        help='Number of batches')

    return parser


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, criterion,
                    args=None):
    model.train()
    print_freq = 20
    # accum_iter = args.accum_iter

    losses = AverageMeter()
    start = time.time()

    optimizer.zero_grad()

    for idx, (images, labels) in enumerate(data_loader):
        bsz = labels.shape[0]
        if args.method == 'LaCViT' or args.method == 'SimCLR':
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            if (idx + 1) % args.accum_iter == 0:
                warmup_learning_rate(args, epoch, idx, len(data_loader), optimizer)

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        elif args.method == "Npair":
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            if (idx + 1) % args.accum_iter == 0:
                warmup_learning_rate(args, epoch, idx, len(data_loader), optimizer)
            features = model(images)


        if args.method == 'LaCViT' or args.method == "Npair":
            loss = criterion(features, labels)
        elif args.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(args.method))

        loss = loss / args.accum_iter
        loss.backward()

        # update metric
        losses.update(loss.item(), bsz)

        if (idx+1) % args.accum_iter == 0:
            # SGD
            # print(loss_accu)
            optimizer.step()
            optimizer.zero_grad()

        if args.distributed:
            torch.cuda.synchronize()
        # measure elapsed time
        end = time.time()
        date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        # print info

        if (idx + 1) % print_freq == 0 : #and idx % args.accum_iter == 0
            log_info = 'Train: [{0}][{1}/{2}]\t' \
                       'lr {lr:.8f} \t' \
                       'loss {loss:.5f}  \t' \
                       'loss_avg({loss_avg: .5f}) \t' \
                       ' {showtime} \n'.format(
                epoch, idx + 1, len(data_loader),
                lr=optimizer.param_groups[0]["lr"], loss=losses.val*args.accum_iter, loss_avg=losses.avg*args.accum_iter,
                showtime=date_time)
            print(log_info)
            sys.stdout.flush()
            with open(args.log_dir + "/cl_mae_log" + str(args.trial) + ".txt", 'a', encoding="utf-8") as f:
                f.write(log_info)
                f.flush()

    return losses.avg


def main(args):
    # prepare the distributed training
    # fix the seed for reproducibility
    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # device = torch.device(args.device)
    if args.distributed:
        NUM_GPUS = torch.cuda.device_count()
        ngpus_per_node = NUM_GPUS
        NUM_CPUS = torch.multiprocessing.cpu_count()
        if NUM_GPUS == 0:
            raise ValueError("you need gpus!")
        if NUM_GPUS > 1:
            args.world_size = misc.get_world_size()
            print(f"World size is {args.world_size}")
        print(f"For this experiment, we are using {NUM_GPUS} GPUs, and {NUM_CPUS} CPUs")
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))
        args.world_size = NUM_GPUS * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        print("{}".format(args).replace(', ', ',\n'))
        main_worker(args.local_rank, args)


def main_worker(rank, args):
    global best_loss
    args.gpu = rank
    print("Currently using GPU: {} for training".format(args.gpu))

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + rank
        print(f"Initialising rank {args.local_rank}")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
        misc.setup_for_distributed(args.local_rank == 0)  # This function disables printing when not in master process

    if misc.is_main_process() and args.wandb_enable:
        config = {
            'lr': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        }
        if args.tag is not None:
            wandb.init(project=args.project, notes=args.note + args.dataset,
                    tags=["LaCViT", args.dataset, args.method, args.tag],
                    config=config, save_code=True #, mode="offline"
                    )
        else:
            wandb.init(project=args.project, notes=args.note + args.dataset,
                    tags=["LaCViT", args.dataset, args.method],
                    config=config, save_code=True #, mode="offline"
                    )

    # build dataset
    if args.distributed:
        if args.load_data_memory == True:
            dataset_train = set_loader_memory(args)
        else:
            dataset_train = set_loader(args)
    else:
        if args.load_data_memory == True:
            data_loader_train = set_loader_memory(args)
        else:
            data_loader_train = set_loader(args)


    model, criterion = set_model(args, project_flag=True)
    optimizer = set_optimizer(args, model)


    if args.resume is not None:
        print("Load model from %s" % args.resume)
        ckpt = torch.load("%s" % args.resume, map_location='cpu')
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        msg = model.load_state_dict(state_dict)
        print("Loading LaCViT model : \n ", msg)
        print("Sucessfully load LaCViTNet weight.")

    # prepare the distributed training
    if args.distributed:  # :
        num_tasks = misc.get_world_size()
        if num_tasks > 1:
            print(f"Distributed training initialised successfully, and the Rank is {num_tasks}")
        global_rank = misc.get_rank()
        print(f"Global rank is{global_rank}")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        # build dataloaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # log training detail
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
            # print("Model = %s" % str(model_without_ddp))
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        # raise ValueError("Distributed training disable!")
        # torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print(f"The effective batch size is {eff_batch_size}")
    print("base lr: %.5e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.5e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print(f"Start training for {args.epochs} epochs")

    # log data into wandb
    if misc.is_main_process() and args.wandb_enable:
        wandb.config.update(args)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.resume and args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)

        time1 = time.time()
        loss = train_one_epoch(
            model, data_loader_train,
            optimizer, epoch, criterion,
            args=args
        )
        print(f"Epoch {epoch} avg loss is {loss}")
        time2 = time.time()

        print('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            if args.distributed:
                if rank == 0:
                    save_model(model.module, optimizer, args, epoch, save_file)
            else:
                save_model(model, optimizer, args, epoch, save_file)

        # no impact on the performance
        if loss < best_loss:
            best_loss = loss #.detach().cpu()
            if args.distributed:
                if rank == 0:
                    save_file = os.path.join(
                        args.save_folder, args.dataset + 'best.pth')
                    save_model(model.module, optimizer, args, epoch, save_file)
                    save_file = os.path.join(
                    args.exchange_path, args.dataset + 'best.pth')
                    save_model(model.module, optimizer, args, args.epochs, save_file)
            else:
                save_file = os.path.join(
                        args.save_folder, args.dataset + 'best.pth')
                save_model(model, optimizer, args, epoch, save_file)
                save_file = os.path.join(
                        args.exchange_path, args.dataset + 'best.pth')
                save_model(model, optimizer, args, args.epochs, save_file)

        log_stats = {
            'Epoch': epoch, 'Loss': loss, 'Lr': optimizer.param_groups[0]['lr'],
            'Time': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

        if args.output_dir and misc.is_main_process():
            if args.wandb_enable:
                wandb.log({"loss": loss, "lr": optimizer.param_groups[0]["lr"]})
                # wandb.run.summary["best_loss"] = loss
                if args.distributed:
                    wandb.watch(model.module, log='all', log_freq=13000)
                else:
                    wandb.watch(model, log='all', log_freq=13000)
            with open(args.log_dir + "/cl_mae_log" + str(args.trial) + ".txt", mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))

    other_ouput = {"Training parameters": "{}".format(args).replace(', ', ',\n'),
                   'Training time': "{}".format(total_time_str),
                   'Lowest training loss': best_loss}
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(other_ouput) + "\n")

    # save the last model
    if args.distributed:
        if rank == 0:
            save_file = os.path.join(
                args.save_folder, 'last.pth')
            save_model(model.module, optimizer, args, args.epochs, save_file)
            save_file = os.path.join(
                args.exchange_path, args.dataset + 'last.pth')
            save_model(model.module, optimizer, args, args.epochs, save_file)
    else:
        save_file = os.path.join(
            args.save_folder, 'last.pth')
        save_model(model, optimizer, args, args.epochs, save_file)
        save_file = os.path.join(
            args.exchange_path, args.dataset + 'last.pth')
        save_model(model, optimizer, args, args.epochs, save_file)
    if args.distributed and misc.is_main_process() and args.wandb_enable:
        wandb.finish()

    # clean up
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # warm-up for large-batch training,
    args.model_save_path = './saved_model/{}/{}_models'.format(args.method, args.dataset)
    args.tb_save_path = './saved_model/{}/{}_tensorboard'.format(args.method, args.dataset)
    # args.exchange_path = './exchange/'
    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
        format(args.method, args.dataset, args.model, args.lr,
               args.weight_decay, args.batch_size, args.temp, args.trial)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.weight_decay ** 3)
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.tb_folder = os.path.join(args.tb_save_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    if not os.path.isdir(args.exchange_path):
        os.makedirs(args.exchange_path)

    args.save_folder = os.path.join(args.model_save_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    print(f"before {args.wandb_enable}")
    args = ModelWrapper.read_and_insert_args(args)
    print(f"after {args.wandb_enable}")

    if args.n_cls is None:
        if args.dataset == 'cifar10':
            args.n_cls = 10
        elif args.dataset == 'cifar100':
            args.n_cls = 100
        elif args.dataset == 'imagenet':
            args.n_cls = 1000
        elif args.dataset == 'bird':
            args.n_cls = 200
        else:
            raise ValueError('dataset not supported: {}'.format(args.dataset))

    main(args)
