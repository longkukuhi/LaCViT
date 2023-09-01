from __future__ import print_function

import pandas as pd
import torch.backends.cudnn as cudnn
import math
import numpy as np
import torch
import torch.optim as optim
from utils.losses import LaCViTLoss, NPairLoss
from models.umae_cl import LaCViTNet, ConMAE
from torchvision import transforms, datasets
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataloaders.datasets import BalancedBatchSampler

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        # print(f"warm epcoh is {args.warm_epochs}")
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if opt.method is None:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    elif opt.method == 'LaCViT' or opt.method == 'SimCLR' :
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.method == 'Npair':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_loader(opt):
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
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    if opt.method == 'Npair':
        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             transform=train_transform,
                                             download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                              transform=train_transform,
                                              download=True)
        elif opt.dataset == 'path':
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=train_transform)
        elif opt.dataset == 'imagenet':
            train_dataset = datasets.ImageFolder(root=opt.imagenet_folder + 'train',
                                                 transform=train_transform,
                                                 )
        elif opt.dataset == 'bird':
            train_dataset = datasets.ImageFolder(root='./datasets/cal_bird/CUB_200_2011/images/',
                                                 transform=train_transform,
                                                 )
        elif opt.dataset == 'caltech256':
            train_dataset = datasets.Caltech256(root=opt.data_folder,
                                                transform=train_transform,
                                                download=True)

        elif opt.dataset == 'flowers102':
            train_dataset = datasets.Flowers102(root=opt.data_folder,
                                                transform=train_transform,
                                                download=True)

        elif opt.dataset == 'oxfordpet':
            train_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                                   transform=train_transform,
                                                   download=True)
        else:
            raise ValueError(opt.dataset)
        train_sampler = BalancedBatchSampler(train_dataset, n_classes=opt.batch_n_cls, n_samples=opt.batch_n)

    else:
        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform),
                                             download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)
        elif opt.dataset == 'path':
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                transform=TwoCropTransform(train_transform))
        elif opt.dataset == 'imagenet':
            train_dataset = datasets.ImageFolder(root=opt.imagenet_folder+'train',
                                 transform=TwoCropTransform(train_transform),
                                 )
        elif opt.dataset == 'bird':
            train_dataset = datasets.ImageFolder(root='./datasets/cal_bird/CUB_200_2011/images/',
                                                 transform=TwoCropTransform(train_transform),
                                                 )
        elif opt.dataset == 'caltech256':
            train_dataset = datasets.Caltech256(root=opt.data_folder,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)

        elif opt.dataset == 'flowers102':
            train_dataset = datasets.Flowers102(root=opt.data_folder,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)

        elif opt.dataset == 'oxfordpet':
            train_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)
        else:
            raise ValueError(opt.dataset)
        train_sampler = None




    if opt.train_length != 1:
        train_len = int(len(train_dataset) * opt.train_length)
        # print(f"traing length is {train_len}")
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    if opt.distributed:
        return train_dataset

    if opt.method == 'Npair':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=opt.num_workers, pin_memory=True, batch_sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    print(f"This trianing uses {len(train_loader)*opt.batch_size} data. It is {opt.train_length} of whole dataset.")
    return train_loader

import os
class Cifar100PreGen:
    def __init__(self, path):
        self.all_labels = pd.read_csv(os.path.join(path, 'train_labels.csv'))
        self.train_img_path = os.path.join(path, 'train')

    def __getitem__(self, item):
        img = torch.load(os.path.join(self.train_img_path, f'img_{item}'))
        label = self.all_labels.iloc[item][1]
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.all_labels)


class DatasetIntoMem:
    def __init__(self, data):
        self.nb_samples = len(data)
        self.all_imgs = []
        self.all_labels = []
        for image, label in data:
            self.all_imgs.append(image)
            self.all_labels.append(label)
        self.use_cache = False

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, idx):
        return self.all_imgs[idx], self.all_labels[idx]

    def __len__(self):
        return self.nb_samples

def set_loader_memory(opt):
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
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    elif opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.imagenet_folder+'train',
                             transform=TwoCropTransform(train_transform),
                             )

    else:
        raise ValueError(opt.dataset)

    train_sampler = None

    train_loader = DatasetIntoMem(train_dataset)
    # if opt.train_length != 1:
    #     train_len = int(len(train_dataset) * opt.train_length)
    #     # print(f"traing length is {train_len}")
    #     train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    #
    # if opt.distributed:
    #     return train_dataset
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
    #     num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    #
    # print(f"This trianing uses {len(train_loader)*opt.batch_size} data. It is {opt.train_length} of whole dataset.")
    return train_loader

def set_loader_split(opt, train=True, mode='pretrain'):
    mean = [0.4859, 0.4996, 0.4318]
    std = [0.1750, 0.1739, 0.1859]
    normalize = transforms.Normalize(mean=mean, std=std)



    if mode == 'pretrain':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'linear':
        train_transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            normalize,
        ])

    if opt.dataset == 'bird':
        if mode == 'pretrain':
            dataset = datasets.ImageFolder(root='./datasets/cal_bird/CUB_200_2011/images/',
                                                 transform=TwoCropTransform(train_transform),
                                                 )
        elif mode == 'linear':
            dataset = datasets.ImageFolder(root='./datasets/cal_bird/CUB_200_2011/images/',
                                           transform=train_transform,
                                           )
        else:
            raise NotImplementedError

    if opt.train_length == 1:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset,[10609,1179])
    else:
        train_len = int(10609*opt.train_length)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, 11788-train_len])
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    if train:
        return train_loader
    else:
        return train_loader, val_loader

def set_model(opt, project_flag=True):
    if opt.method == 'LaCViT' or opt.method == 'SimCLR' :
        model = LaCViTNet(opt, name=opt.model, project_flag=project_flag)
        criterion = LaCViTLoss(temperature=opt.temp)
    elif opt.method == 'Npair':
        model = ConMAE(opt, name=opt.model)
        criterion = NPairLoss(l2_reg=opt.l2_reg)
    else:
        print("Does not support this loss")

    # # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    # model = model.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    return model, criterion

def set_loader_probing(opt, loader=False):
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
        train_dataset = datasets.ImageFolder(root='../datasets/cal_bird/CUB_200_2011/images/',
                                             transform=train_transform,
                                             )
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[10609,1179])

    elif opt.dataset == 'caltech256':
        train_dataset = datasets.Caltech256(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Caltech256(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)

    elif opt.dataset == 'flowers102':
        train_dataset = datasets.Flowers102(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Flowers102(root=opt.data_folder,
                                        split='val',
                                        transform=val_transform, download=True)

    elif opt.dataset == 'oxfordpet':
        train_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.OxfordIIITPet(root=opt.data_folder,
                                        split='test',
                                        transform=val_transform, download=True)
    else:
        raise ValueError(opt.dataset)

    if loader == False:
        return train_dataset, val_dataset
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=8, pin_memory=True)

        return train_loader, val_loader


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

