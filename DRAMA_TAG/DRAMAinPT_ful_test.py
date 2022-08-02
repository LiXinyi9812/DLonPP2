#!/usr/bin/env python3

import datetime
import argparse
import logging
import time
import numpy as np
import torch
import math
import os
import sys
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
#import csv_split
#import renameIMG
from pp2_dataset import PP2Dataset, PP2HDF5Dataset
#from itertools import cycle
from torch.utils import tensorboard as tb
from model import create_DRAMA,get_vgg_backbone,fusion_net,get_backbone,get_alexnet_backbone
import utils
from torch.profiler import profile, record_function, ProfilerActivity



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_w = 224
img_h = 224
image_dir = './pp2/images/'
hdf5_path = './pp2_images.hdf5'
torch.set_default_tensor_type(torch.DoubleTensor)
t = datetime.date.today()


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='drama', help='architecture, drama or others')
    parser.add_argument('--img', '--img-folder', default='./pp2/images/', help='image folder or .hdf5 file')
    parser.add_argument('--dataset', default='./pp2/csv/',
                        help='dataset folder (with {train|val|test}.csv of 6 attributes')
    parser.add_argument('--backbone', default='vgg19',
                        help='one of resnet50, alexnet, vgg16, vgg19 (default: vgg19)')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='cuda or cpu, (default: cuda)')
    parser.add_argument('-b', '--batch-size', default=48, type=int, help='images per batch')
    parser.add_argument('--test-batch-size', default=48, type=int,
                        help='images per batch during evaluation (default: same as --batch-size)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run (default: 10)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help=' from which epoch to start training(default: 0)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # parser.add_argument('--lambda-ranking', type=float, default=1.,
    #                     help='regularization parameter in the loss for rscnn')
    # parser.add_argument('--optim', default='sgd', help='optimizer (sgd or adam)')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M', help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--log-freq', default=200, type=int, help='log progress every `log-freq` batches')
    parser.add_argument('--checkpoint-freq', default=1, type=int,
                        help='Save a checkpoint every `checkpoint-freq` epochs')
    parser.add_argument('--output-dir', default='./output_e2_lr_unfreeze', help='path where to save')
    parser.add_argument('--name', default='', help="Prefix name for the output dir (default: '')")
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')  # TODO: change help message when resume is implemented
    parser.add_argument('--backbone-trainable-layers', default=5, type=int,
                         help='number of trainable layers of backbone. (alexnet: 0 or 1; resnet: between 0 and 3)')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('--no-test', action='store_true', help='only test the model at the end')
    parser.add_argument('--profile', action='store_true', help='profile the training loop, during one epoch at most')
    parser.add_argument('--profiler-steps', default=3, type=int, help='profiler active steps (default: 3)')
    parser.add_argument('--profiler-repeat', default=2, type=int, help='number of cycles for the profiler (default: 2)')
    return parser

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def setup_logger(log_file=None):
    # Setup logging to the console
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%m %H:%M:%S')
    # Setup logging to a file
    if log_file is not None:
        h = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        f = logging.Formatter(fmt='[%(asctime)s] %(name)-12s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
        h.setFormatter(f)
        logging.getLogger('').addHandler(h)  # add the handler to the root logger

def time_elapsed(start):
    elapsed = time.time() - start
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    return elapsed_str


def get(args):
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    data_train = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_validation = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_test = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_train = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_test = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_val = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    print('--------------------------star loading data-----------------------------------')
    for attribute in data_train:
        data_train[attribute] = PP2HDF5Dataset(votes_path=args.dataset+str(attribute) + '_train.csv',hdf5_path=hdf5_path)
        data_validation[attribute] = PP2HDF5Dataset(votes_path=args.dataset + str(attribute) + '_val.csv', hdf5_path=hdf5_path)
        data_test[attribute] = PP2HDF5Dataset(votes_path=args.dataset + str(attribute) + '_test.csv', hdf5_path=hdf5_path)

        train_sampler = torch.utils.data.RandomSampler(data_train[attribute])
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        val_sampler = torch.utils.data.SequentialSampler(data_validation[attribute])
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, args.test_batch_size, drop_last=True)

        test_sampler = torch.utils.data.SequentialSampler(data_test[attribute])
        test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, args.test_batch_size, drop_last=True)



        data_loader_train[attribute] = torch.utils.data.DataLoader(data_train[attribute], batch_sampler=train_batch_sampler,
                                                                   num_workers=args.workers,
                                                                   pin_memory=True)
        data_loader_test[attribute] = torch.utils.data.DataLoader(data_test[attribute], batch_sampler=test_batch_sampler,
                                                                  num_workers=args.workers,
                                                                  pin_memory=True)
        data_loader_val[attribute] = torch.utils.data.DataLoader(data_validation[attribute],
                                                                 batch_sampler=val_batch_sampler,
                                                                 num_workers=args.workers,
                                                                 pin_memory=True)

    print('--------------------------finish loading data-----------------------------------')
    num_train = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    num_validation = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    num_test = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    for attribute in num_train:
        num_validation[attribute] = len(data_validation[attribute])
        num_test[attribute] = len(data_test[attribute])
        num_train[attribute] = len(data_train[attribute])
        print(attribute, '(train number of comparison):', num_train[attribute])
        print(attribute, '(val number of comparison):', num_validation[attribute])
        print(attribute, '(test number of comparison):', num_test[attribute])

    return data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test


def evaluate_accuracy(x,y):
    sm = nn.Softmax(dim=1)
    x = sm(x).argmax(dim=1)
    correct = 0
    for i in range(len(x)):
        if x[i]==y[i]:
            correct += 1
    n = y.shape[0]
    accuracy = correct / n
    return accuracy


def main(args):
    # Output dir
    dic = {0: 'safe', 1: 'lively', 2: 'beautiful', 3: 'wealthy', 4: 'depressing', 5: 'boring'}
    current_datetime_str = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    if args.name:
        args.output_dir = os.path.join(args.output_dir, args.name + '_' + current_datetime_str)
    else:
        args.output_dir = os.path.join(args.output_dir, current_datetime_str)
    os.makedirs(args.output_dir)

    # Setup the logger
    log_file = None
    if args.output_dir is not None:
        log_file = os.path.join(args.output_dir, 'logs.txt')
        print('Logs saved in {}'.format(log_file))
    setup_logger(log_file)
    logger = logging.getLogger('main')
    logger.info(args)

    # Setup tensorboard
    logger.debug('Setting up tensorboard')
    tb_writer = None
    if log_file is not None:
        tb_dir = os.path.join(args.output_dir, 'runs')
        tb_writer = tb.SummaryWriter(log_dir=tb_dir)

    #Setup device
    device = torch.device(args.device)

    logger.debug('Prepare data')
    data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test = get(args)#num is the number of training data
    max_iterations = max(num_train.values())//args.batch_size
    print('max_iterations',max_iterations)
    data_loader_train_final = zip(data_loader_train['safe'], \
                                  cycle(data_loader_train['lively']), \
                                  cycle(data_loader_train['beautiful']), \
                                  cycle(data_loader_train['wealthy']), \
                                  cycle(data_loader_train['depressing']), \
                                  cycle(data_loader_train['boring']))

    logger.debug('Creating model,optimizer and lr_scheduler')
    model_drama = create_DRAMA(device)
    backbone = get_vgg_backbone(19, args.backbone_trainable_layers)
    model_vgg = fusion_net(backbone)
    model_vgg.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_vgg = optim.SGD(model_vgg.parameters(), lr=args.lr, momentum=0.5)
    optimizer_drama = []
    lr_scheduler_drama = []
    n_iter = 0
    for net in model_drama:
        optimizer_drama.append(optim.Adam(net.parameters(), lr=args.lr))
    # Can divide the lr by 10 up to 4 times when the loss is not improving
    for optimizer in optimizer_drama:
        lr_scheduler_drama.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=args.lr / 10000,
                                                                  patience=2))
    lr_scheduler_vgg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_vgg, factor=0.1, min_lr=args.lr / 10000,
                                                              patience=2)


    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        for k in range(6):
            cp = torch.load(args.resume[k])
            model_drama[k].load_state_dict(cp['model_drama_{}'.format(dic[k])])
            optimizer_drama[k].load_state_dict(cp['optimizer_drama_{}'.format(dic[k])])
            lr_scheduler_drama[k].load_state_dict(cp['lr_scheduler_drama_{}'.format(dic[k])])
            # TODO: args.start_epoch = args.epoch + 1 and change the start range in the training loop. Also update n_iter
        model_vgg.load_state_dict(cp['model_vgg'])
        optimizer_vgg.load_state_dict(cp['optimizer_vgg'])
        lr_scheduler_vgg.load_state_dict(cp['lr_scheduler_vgg'])
        args.start_epoch = cp['epoch']
        n_iter = cp['n_iter']


    if args.test_only:
        for k in range(6):
            header = 'Test:'+dic[k]
            test_loss, test_acc = evaluate(dic,k,model_vgg, model_drama, criterion, data_loader_test[dic[k]], device, args.log_freq, header)
            logger.info('test loss: {} - test accuracy: {}'.format(test_loss, test_acc))
            return
    logger.info('Trainable vgg parameters: {}'.format(sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)))
    for k in range(6):
        logger.info('Trainable {} task specific parameters: {}'.format(dic[k],sum(p.numel() for p in model_drama[k].parameters() if p.requires_grad)))


    if args.profile:
        n_iter_profiling = (1 + 1 + args.profiler_steps) * args.profiler_repeat  # (wait + warmup + active) * repeat
        logger.info('Start profiling the training loop for {} iterations'.format(n_iter_profiling))
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profiler_steps,
                                                 repeat=args.profiler_repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.output_dir, 'profiler')),
                record_shapes=False,
                with_stack=False
        ) as prof:
            train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama,lr_scheduler_vgg,lr_scheduler_drama,
                            criterion, data_loader_train_final,  device, 0, 0, tb_writer, args.log_freq ,dic,prof, n_iter_profiling)
    else:
        logger.info('Start training for {} epochs'.format(args.epochs))
        train(max_iterations,args, n_iter,criterion, data_loader_train_final, data_loader_test, data_loader_val,
              device, logger, model_vgg, model_drama,optimizer_vgg, optimizer_drama, lr_scheduler_vgg, lr_scheduler_drama, tb_writer)






def train(max_iterations,args, n_iter,criterion, data_loader_train_final, data_loader_test, data_loader_val,
          device, logger, model_vgg, model_drama,optimizer_vgg, optimizer_drama, lr_scheduler_vgg, lr_scheduler_drama, tb_writer):
    start_time = time.time()
    dic = {0: 'safe', 1: 'lively', 2: 'beautiful', 3: 'wealthy', 4: 'depressing', 5: 'boring'}
    for epoch in range(args.start_epoch+1,args.start_epoch+1+args.epochs):
        n_iter += train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama,lr_scheduler_vgg,lr_scheduler_drama,
                                  criterion, data_loader_train_final, device,epoch, n_iter, tb_writer,args.log_freq,dic)

        #evaluation
        for k in range(6):
            header = 'Val:'+dic[k]
            val_loss, val_accuracy = evaluate(dic,k,model_vgg, model_drama[k], criterion, data_loader_val[dic[k]], device, args.log_freq, header)
            lr_scheduler_vgg.step(val_loss)
            lr_scheduler_drama[k].step(val_loss)
            # if ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == (args.epochs - 1):
            #     cp = {
            #         'model_vgg': model_vgg.state_dict(),
            #         'model_drama_{}'.format(dic[k]): model_drama[k].state_dict(),
            #         'optimizer_vgg': optimizer_vgg.state_dict(),
            #         'optimizer_drama_{}'.format(dic[k]): optimizer_drama[k].state_dict(),
            #         'lr_scheduler_vgg': lr_scheduler_vgg.state_dict(),
            #         'lr_scheduler_drama_{}'.format(dic[k]): lr_scheduler_drama[k].state_dict(),
            #         'args': args,
            #         'epoch': epoch,
            #         'n_iter': n_iter
            #     }
            #     torch.save(cp, os.path.join(args.output_dir, 'model_{}_{}.pt'.format(epoch,dic[k])))
            #     logger.info('[Epoch {}_{}] Checkpoint saved'.format(epoch,dic[k]))
            if tb_writer is not None:
                tb_writer.add_scalar('lr_drama_{}'.format(dic[k]), optimizer_drama[k].param_groups[0]['lr'], n_iter)
                tb_writer.add_scalar('val/loss_{}'.format(dic[k]), val_loss, n_iter)
                tb_writer.add_scalar('val/accuracy_{}'.format(dic[k]), val_accuracy, n_iter)

            if not args.no_test or epoch == (args.epochs - 1):
                header = 'Test:'+dic[k]
                test_loss, test_acc = evaluate(dic,k,model_vgg,model_drama[k], criterion, data_loader_test[dic[k]], device, args.log_freq,header)
                if tb_writer is not None:
                    tb_writer.add_scalar('test/loss_{}'.format(dic[k]), test_loss, n_iter)
                    tb_writer.add_scalar('test/accuracy_{}'.format(dic[k]), test_acc, n_iter)

        tb_writer.add_scalar('lr_vgg', optimizer_vgg.param_groups[0]['lr'], n_iter)

        if ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == (args.epochs - 1):
            cp = {
                'model_vgg': model_vgg.state_dict(),
                'model_drama_safe': model_drama[0].state_dict(),
                'model_drama_lively': model_drama[1].state_dict(),
                'model_drama_beautiful': model_drama[2].state_dict(),
                'model_drama_wealthy': model_drama[3].state_dict(),
                'model_drama_depressing': model_drama[4].state_dict(),
                'model_drama_boring': model_drama[5].state_dict(),
                'optimizer_vgg': optimizer_vgg.state_dict(),
                'optimizer_drama_safe': optimizer_drama[0].state_dict(),
                'optimizer_drama_lively': optimizer_drama[1].state_dict(),
                'optimizer_drama_beautiful': optimizer_drama[2].state_dict(),
                'optimizer_drama_wealthy': optimizer_drama[3].state_dict(),
                'optimizer_drama_depressing': optimizer_drama[4].state_dict(),
                'optimizer_drama_boring': optimizer_drama[5].state_dict(),
                'lr_scheduler_vgg': lr_scheduler_vgg.state_dict(),
                'lr_scheduler_drama_safe': lr_scheduler_drama[0].state_dict(),
                'lr_scheduler_drama_lively': lr_scheduler_drama[1].state_dict(),
                'lr_scheduler_drama_beautiful': lr_scheduler_drama[2].state_dict(),
                'lr_scheduler_drama_wealthy': lr_scheduler_drama[3].state_dict(),
                'lr_scheduler_drama_depressing': lr_scheduler_drama[4].state_dict(),
                'lr_scheduler_drama_boring': lr_scheduler_drama[5].state_dict(),
                'args': args,
                'epoch': epoch,
                'n_iter': n_iter
            }
            torch.save(cp, os.path.join(args.output_dir, 'final_model_{}.pt'.format(epoch)))
            logger.info('[Epoch {}] Checkpoint saved'.format(epoch))

    logger.info('Training time (total) {}'.format(time_elapsed(start_time)))


@torch.no_grad()
def evaluate(dic,k,model_vgg, model_drama, criterion, data_loader, device, log_freq,header):

    # TODO: evaluate with batch_size > 1 for faster computation
    """Return the mean loss and accuracy of the model on the dataset"""
    logger = logging.getLogger('evaluator')
    model_vgg.eval()
    model_drama.eval()
    metric_logger = utils.MetricLogger(logger=logger)
    metric_logger.add_meter('acc_{}'.format(dic[k]), utils.SmoothedValue(fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_{}'.format(dic[k]), utils.SmoothedValue(fmt='{global_avg:.4f}'))

    total_loss, n_samples = 0,0

    for img1, img2, target in metric_logger.log_every(data_loader, log_freq, header):
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        batch_size = img1.size(0)
        n_samples += batch_size
        feature = model_vgg(img1, img2,batch_size)
        output,w = model_drama(feature)
        loss=criterion(output, target.long())  # why +10*nd.mean(w**2)
        loss = loss + 10 * torch.mean(torch.mul(w, w))
        total_loss +=loss.item()*batch_size
        acc_tmp = evaluate_accuracy(output, target)
        metric_logger.meters['acc_{}'.format(dic[k])].update(acc_tmp, n=batch_size)
        metric_logger.meters['loss_{}'.format(dic[k])].update(loss.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    logger.info(header + ' ' + str(metric_logger))
    return total_loss / n_samples, metric_logger.meters['acc_{}'.format(dic[k])].global_avg


def train_one_attribute(k,d,device,model_vgg,model_drama,optimizer_vgg,optimizer_drama,
                        criterion,logger,metric_logger,tb_writer,profiler,n_iter,n_iter_profiling,dic):
    img1, img2, target = d
    img1, img2, target = img1.to(device), img2.to(device), target.to(device)
    batch_size = img1.size(0)
    optimizer_drama[k].zero_grad()
    optimizer_vgg.zero_grad()


    feature = model_vgg(img1, img2,batch_size)
    output, w = model_drama[k](feature)
    loss = criterion(output, target.long()) + 10 * torch.mean(torch.mul(w, w))
    if not math.isfinite(loss):
        logger.info('Loss is {}, stopping training'.format(loss))
        sys.exit(1)

    loss.backward()
    optimizer_vgg.step()
    optimizer_drama[k].step()

    metric_logger.meters['loss_{}'.format(dic[k])].update(value=loss.item(), n=batch_size)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_{}'.format(dic[k]), loss, n_iter)
    if profiler is not None:
        profiler.step()
        #if n_iter >= n_iter_profiling:
        #   break
    k = k+1
    return k


def train_one_iteration(epoch,n_iter,d0,d1,d2,d3,d4,d5,device,optimizer_drama,optimizer_vgg,lr_scheduler_vgg,lr_scheduler_drama,
                        model_vgg,model_drama,criterion,logger, metric_logger,tb_writer,n_iter_profiling,profiler,dic):
    n_iter += 1
    k = 0
    for d in d0,d1,d2,d3,d4,d5 :
        k = train_one_attribute(k,d,device,model_vgg,model_drama,optimizer_vgg,optimizer_drama,
                        criterion,logger,metric_logger,tb_writer,profiler,n_iter,n_iter_profiling,dic)
        if profiler is not None:
            if n_iter >= n_iter_profiling:
                break

    if n_iter%1000 == 0:
        # for k in range(6):
        #     cp = {
        #         'model_vgg': model_vgg.state_dict(),
        #         'model_drama_{}'.format(dic[k]): model_drama[k].state_dict(),
        #         'optimizer_vgg': optimizer_vgg.state_dict(),
        #         'optimizer_drama_{}'.format(dic[k]): optimizer_drama[k].state_dict(),
        #         'lr_scheduler_vgg': lr_scheduler_vgg.state_dict(),
        #         'lr_scheduler_drama_{}'.format(dic[k]): lr_scheduler_drama[k].state_dict(),
        #         'args': args,
        #         'epoch': epoch,
        #         'n_iter': n_iter
        #     }

        cp = {
            'model_vgg': model_vgg.state_dict(),
            'model_drama_safe': model_drama[0].state_dict(),
            'model_drama_lively': model_drama[1].state_dict(),
            'model_drama_beautiful': model_drama[2].state_dict(),
            'model_drama_wealthy': model_drama[3].state_dict(),
            'model_drama_depressing': model_drama[4].state_dict(),
            'model_drama_boring': model_drama[5].state_dict(),
            'optimizer_vgg': optimizer_vgg.state_dict(),
            'optimizer_drama_safe': optimizer_drama[0].state_dict(),
            'optimizer_drama_lively': optimizer_drama[1].state_dict(),
            'optimizer_drama_beautiful': optimizer_drama[2].state_dict(),
            'optimizer_drama_wealthy': optimizer_drama[3].state_dict(),
            'optimizer_drama_depressing': optimizer_drama[4].state_dict(),
            'optimizer_drama_boring': optimizer_drama[5].state_dict(),
            'lr_scheduler_vgg': lr_scheduler_vgg.state_dict(),
            'lr_scheduler_drama_safe': lr_scheduler_drama[0].state_dict(),
            'lr_scheduler_drama_lively': lr_scheduler_drama[1].state_dict(),
            'lr_scheduler_drama_beautiful': lr_scheduler_drama[2].state_dict(),
            'lr_scheduler_drama_wealthy': lr_scheduler_drama[3].state_dict(),
            'lr_scheduler_drama_depressing': lr_scheduler_drama[4].state_dict(),
            'lr_scheduler_drama_boring': lr_scheduler_drama[5].state_dict(),
            'args': args,
            'epoch': epoch,
            'n_iter': n_iter
        }
        torch.save(cp, os.path.join(args.output_dir, 'model_{}_{}.pt'.format(epoch,n_iter)))
        logger.info('[Epoch {}_{}] Checkpoint saved'.format(epoch,n_iter))

    return n_iter


def train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama, lr_scheduler_vgg,lr_scheduler_drama,criterion, data_loader_train_final, device, epoch,
                    n_iter, tb_writer, log_freq, dic,profiler=None,n_iter_profiling=None):
    logger = logging.getLogger('trainer')
    model_vgg.train()
    for model in model_drama:
        model.train()
    metric_logger = utils.MetricLogger(logger=logger)
    #metric_logger.add_meter('acc', utils.SmoothedValue(fmt='{global_avg:.4f}'))


    header = 'Epoch: [{}]'.format(epoch)
    for (d0,d1,d2,d3,d4,d5) in metric_logger.log_every_train(max_iterations,data_loader_train_final, log_freq, header):
        n_iter = train_one_iteration(epoch,n_iter, d0,d1,d2,d3,d4,d5, device, optimizer_drama, optimizer_vgg,lr_scheduler_vgg,lr_scheduler_drama,
                            model_vgg, model_drama, criterion, logger, metric_logger, tb_writer, n_iter_profiling,profiler,dic)
        print('---------------------{} iteration finished----------------------------'.format(n_iter))
    #print_stats()
    return n_iter

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
