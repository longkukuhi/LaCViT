from __future__ import print_function
import os
import sys
import argparse
import time
import math
import torch
import torch.backends.cudnn as cudnn
import wandb
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from utils.util import set_optimizer, save_model, set_loader_probing
from models.umae_cl import SupCEMAE
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=30,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--seed', default=None, type=int)

    # optimization
    parser.add_argument('--lr', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mae_cl_base')
    parser.add_argument('--vit_model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--mae_chkpt_dir', default='./checkpoints/mae_finetuned_vit_base.pth', type=str,
                        help='checkpoint path for mae model')
    parser.add_argument('--n_cls', default=None, type=int,
                        help='Number of classes')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--wandb_enable', action='store_true',
                        help='enable wandb to record')
    parser.add_argument('--project', type=str, default='Test',
                         help='project name for wandb')
    parser.add_argument('--note', type=str, default=' ',
                         help='note for wandb')
    parser.add_argument('--tag', type=str, default=None,
                         help='Tag for wandb')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--data2vec_chkpt', default='./checkpoints/data2vec_base.pth', type=str,
                        help='checkpoint path for data2vec model')
    parser.add_argument('--simmim_chkpt', default=None, type=str,
                        help='checkpoint path for simmim model')
    parser.add_argument('--method', type=str, default=None, help='choose method')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './saved_model/ce/{}_models'.format(opt.dataset)



    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.lr, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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



    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

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

    return opt


def set_model(opt):
    model = SupCEMAE(opt, name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    optimizer.zero_grad()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)


        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # SGD
        loss = loss / opt.accum_iter
        loss.backward()

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        if (idx + 1) % opt.accum_iter == 0:
            # SGD
            # print(loss_accu)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.6f}\t'
                  'loss {loss:.3f} ({loss_avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    lr=optimizer.param_groups[0]["lr"],
                   data_time=data_time, loss=losses.val* opt.accum_iter,
                   loss_avg=losses.avg* opt.accum_iter, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():


    best_acc = 0
    opt = parse_option()
    print(opt)

    # fix the seed for reproducibility
    if opt.seed is not None:
        seed = opt.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # build data loader
    train_loader, val_loader = set_loader_probing(opt, loader=True)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    if opt.resume is not None:
        print("Load model from %s" % opt.resume)
        state_dict = torch.load("%s" % opt.resume)
        msg = model.load_state_dict(state_dict['model'])
        print("Sucessfully load SupConNet weight.")


    if  opt.wandb_enable:
        config = {
            'lr': opt.lr,
            'epochs': opt.epochs,
            'batch_size': opt.batch_size,
        }
        if opt.tag is not None:
            wandb.init(project=opt.project, notes=opt.note + opt.dataset,
                    tags=["LaCViT", opt.dataset, opt.tag],
                    config=config, save_code=True #, mode="offline"
                    )
        else:
            wandb.init(project=opt.project, notes=opt.note + opt.dataset,
                    tags=["LaCViT", opt.dataset,],
                    config=config, save_code=True #, mode="offline"
                    )

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # # tensorboard logger
        # logger.log_value('train_loss', loss, epoch)
        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        val_loss, val_acc = validate(val_loader, model, criterion, opt)
        # logger.log_value('val_loss', loss, epoch)
        # logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            # save the last model
            save_file = os.path.join(
                opt.save_folder, 'best.pth')
            save_model(model, optimizer, opt, opt.epochs, save_file)

        if opt.wandb_enable:
            wandb.log({"trian_loss":train_loss, "val_loss": val_loss,
                       "train_acc":train_acc, "val_acc": val_acc,
                       "lr": optimizer.param_groups[0]["lr"]})

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
