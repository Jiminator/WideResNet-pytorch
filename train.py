import argparse
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler

from profiler import LayerProfiler
from profiler import TrainingTimer
from wideresnet import WideResNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
# This argument is required for distributed training.
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', type=int, default=0,
                    help='Local rank. Necessary for using the torch.distributed.launch utility.')
parser.add_argument('--num_workers', default=4,
                    type=int, help='number of workers for data loader.')
parser.set_defaults(augment=True)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Initialize distributed training if applicable.
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1


    # Override local_rank from environment variable if present.
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.distributed:
        # torch.cuda.set_device(args.local_rank)
        # dist.init_process_group(backend='nccl', init_method='env://')
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, init_method='env://')
        args.device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(args.device)
        
    # if args.distributed:
    #     if 'SLURM_PROCID' in os.environ:  # for slurm scheduler
    #         args.global_rank = int(os.environ['SLURM_PROCID'])
    #         args.local_rank = args.global_rank % torch.cuda.device_count()
    #         dist.init_process_group(backend=args.dist_backend,
    #                                 world_size=args.world_size, rank=args.global_rank)
    #     else:
    #         raise ValueError("SLURM_PROCID not found in env.")
    #     args.device = torch.device(f"cuda:{args.local_rank}")
    #     torch.cuda.set_device(args.device)
    
    # Only rank 0 will log to TensorBoard.
    if args.tensorboard and (not args.distributed or (args.distributed and args.local_rank == 0)):
        configure("runs/%s" % (args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Create datasets
    assert args.dataset in ['cifar10', 'cifar100']
    train_dataset = datasets.__dict__[args.dataset.upper()](
        '../data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.__dict__[args.dataset.upper()](
        '../data', train=False, transform=transform_test)

    # Create Distributed Samplers if using distributed training.
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)

    # create model
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    if not args.distributed or (args.distributed and args.local_rank == 0):
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda(args.device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if not args.distributed or (args.distributed and args.local_rank == 0):
                print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.distributed or (args.distributed and args.local_rank == 0):
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if not args.distributed or (args.distributed and args.local_rank == 0):
                print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    profiler = LayerProfiler(model, save_dir=f'runs/{args.name}/profile_data')
    timer = TrainingTimer(save_dir=f'runs/{args.name}/timer_data')
    # cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, scheduler, epoch, timer)
        # Only rank 0 saves profiling and timer data.
        if not args.distributed or (args.distributed and args.local_rank == 0):
            profiler.save_profile_data(epoch)
            timer.save_statistics(epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint (only on rank 0)
        if not args.distributed or (args.distributed and args.local_rank == 0):
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    if not args.distributed or (args.distributed and args.local_rank == 0):
        profiler.close()
        print('Best accuracy: ', best_prec1)

    # Cleanup the distributed process group.
    cleanup()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, timer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        timer.start('total_step_time')
        timer.start('batch_generator_time')
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        timer.stop('batch_generator_time')

        timer.start('forward_backward_time')
        output = model(input)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        timer.stop('forward_backward_time')

        timer.start('optimizer_time')
        optimizer.step()
        scheduler.step()
        timer.stop('optimizer_time')
        
        timer.stop('total_step_time')

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and (not args.distributed or (args.distributed and args.local_rank == 0)):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    if args.tensorboard and (not args.distributed or (args.distributed and args.local_rank == 0)):
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and (not args.distributed or (args.distributed and args.local_rank == 0)):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    if not args.distributed or (args.distributed and args.local_rank == 0):
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        if args.tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', top1.avg, epoch)
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk (only rank 0 saves the checkpoint)"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

def cleanup():
    """Cleanup the distributed process group."""
    if getattr(args, 'distributed', False):
        dist.destroy_process_group()

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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # os.environ["NCCL_SOCKET_IFNAME"] = "ib"
    main()