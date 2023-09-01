import argparse
import datetime
import json
import os
import time
import warnings
import sys
import math
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
# own-defined library
import utils.misc as misc
from utils.util import AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
from utils.util import set_model, set_loader, accuracy
import torch.backends.cudnn as cudnn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms, datasets
from models.umae_cl import LaCViTNet, LinearClassifier, ConMAE
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('Linear training', add_help=False)

    # params
    parser.add_argument('--print_freq', type=int, default=6,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--seed', default=None, type=int)

    # optimization
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,7,10',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

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

    # other setting
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--log_dir', default='output_dir', type=str,
                        help='GPU id to use.')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--mae_chkpt_dir', default=None, type=str,
                        help='checkpoint path for mae model')
    parser.add_argument('--data2vec_chkpt', default='./checkpoints/data2vec_base.pth', type=str,
                        help='checkpoint path for data2vec model')
    parser.add_argument('--train_length', type=float, default=1,
                        help='length of training set')
    parser.add_argument('--method', type=str, default='LaCViT',
                        choices=['LaCViT', 'SimCLR', 'Npair'], help='choose method')
    parser.add_argument('--n_cls', default=None, type=int,
                        help='Number of classes')
    #wandb paras
    parser.add_argument('--wandb_enable', action='store_true',
                        help='enable wandb to record')
    parser.add_argument('--project', type=str, default='Test',
                         help='project name for wandb')
    parser.add_argument('--note', type=str, default=' ',
                         help='note for wandb')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--tag', type=str, default=None,
                         help='Tag for wandb')

    # model dataset
    parser.add_argument('--model', type=str, default='mae_cl_base')
    parser.add_argument('--vit_model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--imagenet_folder', type=str, default='./datasets/imagenet/imagenet/data/CLS-LOC/',
                        help='path to custom dataset')
    parser.add_argument('--simmim_chkpt', default=None, type=str,
                        help='checkpoint path for simmim model')

    # distributed training parameters

    parser.add_argument('--distributed', default=False, type=bool,
                        help='whether distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_cluster', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    opt = parser.parse_args()
    opt.model_path = './saved_model/{}_linear/{}_models'.format(opt.method, opt.dataset)

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'. \
        format(opt.dataset, opt.model, opt.lr, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.lr * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.lr - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.lr

    if opt.n_cls is None:
        if opt.dataset == 'cifar10':
            opt.n_cls = 10
        elif opt.dataset == 'cifar100':
            opt.n_cls = 100
        elif opt.dataset == 'imagenet':
            opt.n_cls = 1000
        elif opt.dataset == 'bird':
            opt.n_cls = 200
        else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt,):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif opt.dataset == 'bird':
        mean = [0.4859, 0.4996, 0.4318]
        std = [0.1750, 0.1739, 0.1859]
    elif opt.dataset == 'caltech256':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.dataset == 'flowers102':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.dataset == 'oxfordpet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.imagenet_folder+"train",
                             transform=train_transform,
                             )
        val_dataset = datasets.ImageFolder(root=opt.imagenet_folder+'val',
                                             transform=train_transform,
                                             )
        if opt.train_length != 1:
            train_len = int(len(train_dataset) * opt.train_length)
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset) - train_len ])
    elif opt.dataset == 'bird':
        train_dataset = datasets.ImageFolder(root='./datasets/cal_bird/CUB_200_2011/images/',
                                             transform=train_transform,
                                             )
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[10609,1179])

    elif opt.dataset == 'caltech256':
        train_dataset = datasets.Caltech256(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Caltech256(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform, download=True)

    elif opt.dataset == 'flowers102':
        train_dataset = datasets.Flowers102(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Flowers102(root=opt.data_folder,
                                        split='test',
                                        transform=val_transform,download=True)

    elif opt.dataset == 'oxfordpet':
        train_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                             split='test', transform=val_transform,download=True)

    else:
        raise ValueError(opt.dataset)


    return train_dataset, val_dataset



def set_model(opt, load_ck=True):
    if opt.method == 'LaCViT' or opt.method == 'SimCLR':
        model = LaCViTNet(opt, name=opt.model)
    elif opt.method == 'Npair':
        model = ConMAE(opt, name=opt.model)

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')

    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    cudnn.benchmark = True

    if load_ck:
        msg = model.load_state_dict(state_dict)
        print("message of loading LaCViTNet model: \n", msg)
    else:
        print("Does not load contrastive learning, LaCViTNet weight.")

    return model, classifier, criterion

def train_one_epoch(data_loader, model, classifier, criterion, optimizer, epoch, args):
    model.eval()
    classifier.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    start = time.time()


    for idx, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(args, epoch, idx, len(data_loader), optimizer)

        # compute loss
        if args.distributed:
            with torch.no_grad():
                features = model.module.encoder(images)
        else:
            with torch.no_grad():
                features = model.encoder(images)

        output = classifier(features)
        # if args.dataset == 'flowers102':
        #     labels = labels - 1
        # print(labels)
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        loss = loss / args.accum_iter
        loss.backward()

        if (idx + 1) % args.accum_iter == 0:
            # SGD
            optimizer.step()
            optimizer.zero_grad()

        if args.distributed:
            torch.cuda.synchronize()

        # measure elapsed time
        end = time.time()

        date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        if (idx + 1) % args.print_freq == 0:
            # print info
            log_info = 'Train: [{0}][{1}/{2}]\t' \
                       'lr {lr:.8f} \t' \
                       'Train_loss {loss.val:.5f} ({loss.avg:.5f}) \t' \
                       'Train_Acc@1 {top1.val:.5f} ({top1.avg:.5f} \t' \
                       '{showtime} \n)'.format(
                epoch, idx + 1, len(data_loader),
                lr=optimizer.param_groups[0]["lr"], loss=losses, top1=top1,
                showtime=date_time)

            print(log_info)
            sys.stdout.flush()
            with open(args.log_dir + f"/cl_mae_linaer_log_{str(args.trial)}.txt", 'a') as f:
                f.write(log_info)
                f.flush()

    return losses.avg, top1.avg

def validate(val_loader, model, classifier, criterion, args):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        start = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if args.distributed:
                output = classifier(model.module.encoder(images))
            else:
                output = classifier(model.encoder(images))
            # if args.dataset == 'flowers102':
            #     labels = labels - 1
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - start)
            # end = time.time()

            if idx % args.print_freq == 0:
                log_info = 'Test: [{0}][{1}/{2}]\t' \
                           'loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                           'Acc@1 {top1.val:.5f} ({top1.avg:.5f}' \
                           '{showtime})'.format(
                    idx, idx + 1, len(val_loader), loss=losses, top1=top1,
                    showtime=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

                print(log_info)
                sys.stdout.flush()
                with open(args.log_dir + f"/cl_mae_linear_{str(args.trial)}_log.log", 'a') as f:
                    f.write(log_info)
                    f.flush()


    print(' * Acc@1 {top1.avg:.5f}'.format(top1=top1))
    return losses.avg, top1.avg

def main(args):
    # prepare the distributed training
    # fix the seed for reproducibility
    if args.seed is not None:
        seed = args.seed + misc.get_rank()
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
    # ************************************************************
    global best_acc
    best_acc = 0
    args.gpu = rank

    # ************************************************************
    # initial distribution training
    print("Currently using GPU: {} for training".format(args.gpu))
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.local_rank = args.local_rank * ngpus_per_node + rank
        print(f"Initialising rank {args.local_rank}")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
        misc.setup_for_distributed(args.local_rank == 0)
    # ************************************************************
    # enable log by wandb
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

    # ************************************************************
    # build dataset
    # build data loader
    # todo: need to modified to handle distributed training
    dataset_train, dataset_val = set_loader(args)
    # build model and criterion
    model, classifier, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, classifier)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    #************************************************************
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
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    # build dataloaders
    # todo: need to modified to handle distributed training
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,  batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=1024,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
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

    # ************************************************************
    # resume from checkpoint
    if args.resume is not None:
        print("Load model from %s" % args.resume)
        state_dict = torch.load("%s" % args.resume)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        msg = model.module.load_state_dict(state_dict['model'])
        print("Loading LaCViTNet model : \n ", msg)
        print("Sucessfully load LaCViTNet weight.")

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    # log data into wandb
    if misc.is_main_process() and args.wandb_enable:
        wandb.config.update(args)

    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)

        time1 = time.time()
        loss, acc = train_one_epoch(data_loader_train, model, classifier, criterion,
                          optimizer, epoch, args)
        time2 = time.time()
        print(f"Epoch {epoch} avg loss is {loss}")
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))


        # eval for one epoch
        val_loss, val_acc = validate(data_loader_val, model, classifier, criterion, args)

        if val_acc > best_acc:
            best_acc = val_acc.detach().cpu().item()
            save_file = os.path.join(
                args.save_folder, 'Best_ckpt_linear_epoch_{epoch}.pth'.format(epoch=epoch))
            if args.distributed and misc.is_main_process():
                if args.wandb_enable:
                    wandb.run.summary["best_accuracy"] = best_acc
            elif not args.distributed:
                if args.wandb_enable:
                    wandb.run.summary["best_accuracy"] = best_acc


            if rank == 0 and args.distributed:
                save_model(model_without_ddp, optimizer, args, epoch, save_file)
            else:
                save_model(model, optimizer, args, epoch, save_file)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_linear_epoch_{epoch}.pth'.format(epoch=epoch))
            if rank == 0 and args.distributed:
                save_model(model_without_ddp, optimizer, args, epoch, save_file)
            else:
                save_model(model, optimizer, args, epoch, save_file)


        log_stats = {
                     'Epoch': epoch,  'val_loss': val_loss, 'val_acc': val_acc.detach().cpu().item(),
                    'time': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

        if args.log_dir:
            if args.distributed and misc.is_main_process():
                if args.wandb_enable:
                    wandb.log({"val_loss": val_loss, "train_loss": loss,
                            "train_acc": acc, "val_acc": val_acc,
                            "lr": optimizer.param_groups[0]["lr"]})
                    wandb.watch(model.module, log='all', log_freq=13000)
            elif not args.distributed:
                if args.wandb_enable:
                    wandb.log({"val_loss": val_loss, "train_loss": loss,
                            "train_acc": acc, "val_acc": val_acc,
                            "lr": optimizer.param_groups[0]["lr"]})
                # wandb.log({"train_loss": loss})
                # wandb.log({"train_acc": acc})
                # wandb.log({"val_acc": val_acc})
                # wandb.log({"lr": optimizer.param_groups[0]["lr"]})
                    wandb.watch(model, log='all', log_freq=13000)

            print(log_stats)
            # if log_writer is not None:
            #     log_writer.flush()
            with open(args.log_dir + f"/cl_mae_linaer_log_{str(args.trial)}.txt", 'a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    print('Best accuracy: {:.4f}'.format(best_acc))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    other_ouput = {"Training parameters": "{}".format(args).replace(', ', ',\n'),
                   'Training time': "{}".format(total_time_str),
                   'best accuracy': best_acc}

    #save model
    if args.log_dir and misc.is_main_process():
        print(other_ouput)
        # if log_writer is not None:
        #     log_writer.flush()
        with open(args.log_dir + f"/cl_mae_linaer_log_{str(args.trial)}.txt", 'a', encoding="utf-8") as f:
            f.write(json.dumps(other_ouput) + "\n")
            f.flush()



    # save the last model
    if args.distributed and rank == 0:
        save_file = os.path.join(
            args.save_folder, 'linear_last.pth')
        save_model(model_without_ddp, optimizer, args, args.epochs, save_file)
    else:
        save_file = os.path.join(
            args.save_folder, 'linear_last.pth')
        save_model(model, optimizer, args, args.epochs, save_file)

    # clean up
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    main(args)

