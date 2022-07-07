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
from torchvision import models
import torchvision
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
#import csv_split
#import renameIMG
from pp2_dataset import PP2Dataset, PP2HDF5Dataset
from itertools import cycle
from torch.utils import tensorboard as tb
from model import create_vgg19, create_DRAMA
import utils



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 48
img_w = 224
img_h = 224
test_iteration = 4000
torch.set_default_tensor_type(torch.DoubleTensor)

image_dir = './pp2/images/'
csv_dir = './pp2/csv/'
outputdir = './output_test'
hdf5_path = './pp2_images.hdf5'
t = datetime.date.today()
workers = 4

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='drama', help='architecture, drama or others')
    parser.add_argument('--img', '--img-folder', default='./pp2/images/', help='image folder or .hdf5 file')
    parser.add_argument('--dataset', default='./pp2/csv/',
                        help='dataset folder (with {train|val|test}.csv of 6 attributes')
    parser.add_argument('--backbone', default='vgg19',
                        help='one of resnet50, alexnet, vgg16, vgg19 (default: vgg19)')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='cuda or cpu, (default: cuda)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='images per batch')
    parser.add_argument('--test-batch-size', default=2, type=int,
                        help='images per batch during evaluation (default: same as --batch-size)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run (default: 10)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # parser.add_argument('--lambda-ranking', type=float, default=1.,
    #                     help='regularization parameter in the loss for rscnn')
    # parser.add_argument('--optim', default='sgd', help='optimizer (sgd or adam)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M', help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--log-freq', default=100, type=int, help='log progress every `log-freq` batches')
    parser.add_argument('--checkpoint-freq', default=1, type=int,
                        help='Save a checkpoint every `checkpoint-freq` epochs')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--name', default='', help="Prefix name for the output dir (default: '')")
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')  # TODO: change help message when resume is implemented
    # parser.add_argument('--backbone-trainable-layers', default=3, type=int,
    #                     help='number of trainable layers of backbone. (alexnet: 0 or 1; resnet: between 0 and 3)')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('--no-test', action='store_true', help='only test the model at the end')
    parser.add_argument('--profile', action='store_true', help='profile the training loop, during one epoch at most')
    parser.add_argument('--profiler-steps', default=3, type=int, help='profiler active steps (default: 3)')
    parser.add_argument('--profiler-repeat', default=2, type=int, help='number of cycles for the profiler (default: 2)')
    return parser

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


# def csv2data(csv_path,dataframe,image_dir):
#     csv_reader = np.loadtxt(csv_path, delimiter='\t', dtype=str)
#     for row in csv_reader:
#         dataframe.append({'dir':[image_dir+str(row[2])+'.JPG', image_dir+str(row[3])+'.JPG'],\
#              'label':0 if row[1]=='left' else 1})
#     return dataframe


def get(csv_dir):
    #csv_reader = np.loadtxt(csv_path,delimiter='\t',dtype=str)
    data = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
    data_train = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_validation = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_test = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_train = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_test = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    data_loader_val = {'safe': [], 'lively': [], 'beautiful': [], 'wealthy': [], 'depressing': [], 'boring': []}
    print('--------------------------star loading data-----------------------------------')
    for attribute in data:
        # data[attribute] = csv2data(csv_dir+str(attribute)+'.csv', data[attribute],image_dir)
        # data_train[attribute] = csv2data(csv_dir+str(attribute)+'_train.csv',data_train[attribute],image_dir)
        # data_validation[attribute] = csv2data(csv_dir+str(attribute)+'_val.csv',data_validation[attribute],image_dir)
        # data_test[attribute] = csv2data(csv_dir + str(attribute) + '_test.csv', data_test[attribute],image_dir)
        data[attribute] = PP2HDF5Dataset(votes_path=csv_dir+str(attribute)+'.csv',hdf5_path=hdf5_path)
        data_train[attribute] = PP2HDF5Dataset(votes_path=csv_dir+str(attribute) + '_train.csv',hdf5_path=hdf5_path)
        data_validation[attribute] = PP2HDF5Dataset(votes_path=csv_dir + str(attribute) + '_val.csv', hdf5_path=hdf5_path)
        data_test[attribute] = PP2HDF5Dataset(votes_path=csv_dir + str(attribute) + '_test.csv', hdf5_path=hdf5_path)

        train_sampler = torch.utils.data.RandomSampler(data_train[attribute])
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

        val_sampler = torch.utils.data.SequentialSampler(data_validation[attribute])
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size, drop_last=False)

        test_sampler = torch.utils.data.SequentialSampler(data_test[attribute])
        test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, batch_size, drop_last=False)



        data_loader_train[attribute] = torch.utils.data.DataLoader(data_train[attribute], batch_sampler=train_batch_sampler,
                                                                   num_workers=workers,
                                                                   pin_memory=True)
        data_loader_test[attribute] = torch.utils.data.DataLoader(data_test[attribute], batch_sampler=test_batch_sampler,
                                                                  num_workers=workers,
                                                                  pin_memory=True)
        data_loader_val[attribute] = torch.utils.data.DataLoader(data_validation[attribute],
                                                                 batch_sampler=val_batch_sampler,
                                                                 num_workers=workers,
                                                                 pin_memory=True)

    print('--------------------------finish loading data-----------------------------------')
    num = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
    num_train = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    num_validation = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    num_test = {'safe': 0, 'lively': 0, 'beautiful': 0, 'wealthy': 0, 'depressing': 0, 'boring': 0}
    for attribute in num:
        num[attribute]= len(data[attribute])
        print(attribute,'(number of comparison):',num[attribute])
        num_validation[attribute] = len(data_validation[attribute])
        num_test[attribute] = len(data_test[attribute])
        num_train[attribute] = len(data_train[attribute])
        print(attribute, '(train number of comparison):', num_train[attribute])
        print(attribute, '(val number of comparison):', num_validation[attribute])
        print(attribute, '(test number of comparison):', num_test[attribute])

    return data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test


def fusion(img_l,img_r,model_vgg):
    left_feature = model_vgg.features(img_l)
    left_feature = model_vgg.avgpool(left_feature)
    right_feature = model_vgg.features(img_r)
    right_feature = model_vgg.avgpool(right_feature)
    left_feature = left_feature.reshape((batch_size, 512, -1))
    left_feature = left_feature.squeeze()
    right_feature = right_feature.reshape((batch_size, 512, -1))
    right_feature = right_feature.squeeze()
    left_feature = left_feature.permute(0, 2, 1)
    join_feature = torch.bmm(left_feature, right_feature)
    feature = join_feature.unsqueeze(dim=1)
    return feature

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

#prepare network
class DRAMA(nn.Module):
    def __init__(self):
        super(DRAMA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 10,kernel_size=3)
        self.relu1 = nn.Softplus()
        self.conv2 = nn.Conv2d(in_channels=10,out_channels= 10,kernel_size=3)
        self.relu2 = nn.Softplus()
        self.conv3 = nn.Conv2d(in_channels=10,out_channels= 10,kernel_size=3)
        self.relu3 = nn.Softplus()
        self.dense = nn.Linear(in_features=10*43*43,out_features=1*2)
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.relu1(x_1)
        x_3 = self.conv2(x_2)
        x_4 = self.relu2(x_3)
        x_5 = self.conv3(x_4)
        x_6 = self.relu3(x_5)
        x_7 = x_6.view(batch_size,10*43*43)
        res = self.dense(x_7)
        return  res,self.dense.weight.data


def plot_loss_accuracy(list_loss,list_train_acc,list_val_acc,epoch):
    for k in list_loss:
        x_l = np.arange(0, epoch, 1)
        y_l = np.array(torch.tensor(list_loss[k],device='cpu'))
        x_ta = np.arange(0, epoch, 1)
        y_ta = np.array(torch.tensor(list_train_acc[k],device='cpu'))
        x_va = np.arange(0, epoch, 1)
        y_va = np.array(torch.tensor(list_val_acc[k],device='cpu'))

        plt.subplot(3, 1, 1)
        plt.plot(x_l, y_l, 'o-')
        plt.title('Train loss of %s' % k)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(3, 1, 2)
        plt.plot(x_ta, y_ta, 'o-')
        plt.title('Train accuracy of %s' % k)
        plt.xlabel('Epoch')
        plt.ylabel('Train accuracy')

        plt.subplot(3, 1, 3)
        plt.plot(x_va, y_va, 'o-')
        plt.title('Validation accuracy of %s' % k)
        plt.xlabel('Epoch')
        plt.ylabel('Validation accuracy')

        plt.tight_layout()
        plt.savefig(outputdir+'Loss_TrainAccuracy_ValidationAccuracy_' + str(k) + '.jpg')
        plt.show()


#preprocess data
# renameIMG.rename(image_dir)
# csv_split.splitAtt(csv_path,csv_dir)
#
# test_rate = 0.35
# val_rate =  0.05/0.6
# for att in ['safe','lively','beautiful','wealthy','depressing','boring']:
#     csv_split.splitTVT(csv_dir+str(att)+'.csv', att, test_rate, val_rate,csv_dir,batch_size)

def main(args):
    # Output dir
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

    device = torch.device(args.device)

    logger.debug('Prepare data')
    data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test = get(csv_dir)#num is the number of training data
    list_val_acc = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
    list_train_acc = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
    list_loss = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
    max_acc = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
    acc_train = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
    max_iterations = max(num_train.values())//batch_size
    print('max_iterations',max_iterations)
    data_loader_train_final = zip(data_loader_train['safe'], \
                                  cycle(data_loader_train['lively']), \
                                  cycle(data_loader_train['beautiful']), \
                                  cycle(data_loader_train['wealthy']), \
                                  cycle(data_loader_train['depressing']), \
                                  cycle(data_loader_train['boring']))

    logger.debug('Creating model')
    model_drama = create_DRAMA(device)
    model_vgg = create_vgg19('./vgg_extractor')
    model_vgg.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_vgg = optim.SGD(model_vgg.parameters(), lr=0.001, momentum=0.5)
    optimizer_drama = []
    for net in model_drama:
        optimizer_drama.append(optim.Adam(net.parameters(), lr=0.001))


    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        cp_vgg = torch.load(args.resume[0])
        model_vgg.load_state_dict(cp_vgg['model'])
        for k in range(6):
            cp_drama = torch.load(args.resume[k])
            model_drama[k].load_state_dict(cp_drama['model'])
            model_drama[k].eval()

    if args.test_only:
        dic = {0: 'safe', 1: 'lively', 2: 'beautiful', 3: 'wealthy', 4: 'depressing', 5: 'boring'}
        for k in range(6):
            header = 'Test:'+dic[k]
            test_loss, test_acc = evaluate(model_vgg, model_drama, criterion, data_loader_test[dic[k]], device, args.log_freq, header)
            logger.info('test loss: {} - test accuracy: {}'.format(test_loss, test_acc))
            return

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
            train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama, criterion, data_loader_train_final,  device, 0, 0, tb_writer, args.log_freq, prof, n_iter_profiling)
    else:
        logger.info('Start training for {} epochs'.format(args.epochs))
        train(max_iterations,args, criterion, data_loader_train_final, data_loader_test, data_loader_val, device, logger, model_vgg, model_drama,optimizer_vgg, optimizer_drama, tb_writer)


def train(max_iterations,args, criterion, data_loader_train_final, data_loader_test, data_loader_val, device, logger, model_vgg, model_drama,optimizer_vgg, optimizer_drama, tb_writer):
    start_time = time.time()
    n_iter = 0
    dic = {0: 'safe', 1: 'lively', 2: 'beautiful', 3: 'wealthy', 4: 'depressing', 5: 'boring'}
    for epoch in range(args.epochs):
        n_iter += train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama, criterion, data_loader_train_final, device,epoch, n_iter, tb_writer,args.log_freq)
        for k in range(6):
            header = 'Val:'+dic[k]
            val_loss, val_accuracy = evaluate(model_vgg, model_drama[k], criterion, data_loader_val[dic[k]], device, args.log_freq, header)


            if ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == (args.epochs - 1):
                cp = {
                    'model_vgg': model_vgg.state_dict(),
                    'model_drama': model_drama[k].state_dict(),
                    'optimizer_vgg': optimizer_vgg.state_dict(),
                    'optimizer_drama': optimizer_drama[k].state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'n_iter': n_iter
                }
                torch.save(cp, os.path.join(args.output_dir, 'model_{}_{}.pt'.format(epoch,dic[k])))
                logger.info('[Epoch {}] Checkpoint saved'.format(epoch)+dic[k])
            if tb_writer is not None:
                tb_writer.add_scalar('lr_vgg'+dic[k], optimizer_vgg.param_groups[0]['lr'], n_iter)
                tb_writer.add_scalar('lr_drama'+dic[k], optimizer_drama[k].param_groups[0]['lr'], n_iter)
                tb_writer.add_scalar('val/loss'+dic[k], val_loss, n_iter)
                tb_writer.add_scalar('val/accuracy'+dic[k], val_accuracy, n_iter)

            if not args.no_test or epoch == (args.epochs - 1):
                header = 'Test:'+dic[k]
                test_loss, test_acc = evaluate(model_vgg,model_drama[k], criterion, data_loader_test[dic[k]], device, args.log_freq,header)
                if tb_writer is not None:
                    tb_writer.add_scalar('test/loss'+dic[k], test_loss, n_iter)
                    tb_writer.add_scalar('test/accuracy'+dic[k], test_acc, n_iter)

    logger.info('Training time (total) {}'.format(time_elapsed(start_time)))

@torch.no_grad()
def evaluate(model_vgg, model_drama, criterion, data_loader, device, log_freq,header):

    # TODO: evaluate with batch_size > 1 for faster computation
    """Return the mean loss and accuracy of the model on the dataset"""
    logger = logging.getLogger('evaluator')
    model_vgg.eval()
    model_drama.eval()
    metric_logger = utils.MetricLogger(logger=logger)
    metric_logger.add_meter('acc', utils.SmoothedValue(fmt='{global_avg:.4f}'))

    total_loss, accuracy, n_samples = 0,0,0

    for img1, img2, target in metric_logger.log_every(data_loader, log_freq, header):
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        print('target_eval', target)
        batch_size = img1.size(0)
        n_samples += batch_size
        feature = fusion(img1, img2, model_vgg)
        output,w = model_drama(feature)
        loss=criterion(output, target.long())  # why +10*nd.mean(w**2)
        loss = loss + 10 * torch.mean(torch.mul(w, w))
        total_loss +=loss.item()*batch_size
        acc_tmp = evaluate_accuracy(output, target)
        # accuracy += acc_tmp
        #acc = (torch.abs(target - output) <= 0.5).sum().item() / batch_size
        metric_logger.meters['acc'].update(acc_tmp, n=batch_size)
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    logger.info(header + ' ' + str(metric_logger))
    return total_loss / n_samples, metric_logger.acc.global_avg

def train_one_epoch(max_iterations,model_vgg,model_drama,  optimizer_vgg, optimizer_drama, criterion, data_loader_train_final, device, epoch,
                    n_iter, tb_writer, log_freq, profiler=None,n_iter_profiling=None):
    dic = {0: 'safe', 1: 'lively', 2: 'beautiful', 3: 'wealthy', 4: 'depressing', 5: 'boring'}
    logger = logging.getLogger('trainer')
    model_vgg.train()
    for model in model_drama:
        model.train()
    metric_logger = utils.MetricLogger(logger=logger)
    header = 'Epoch: [{}]'.format(epoch)
    for img1img2target in metric_logger.log_every_train(max_iterations,data_loader_train_final, log_freq, header):
        n_iter += 1
        for k in range(6):
            img1, img2, target = img1img2target[k]
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            batch_size = img1.size(0)
            optimizer_drama[k].zero_grad()
            optimizer_vgg.zero_grad()
            feature = fusion(img1, img2, model_vgg)
            output, w = model_drama[k](feature)
            loss = criterion(output, target.long()) + 10 * torch.mean(torch.mul(w, w))
            loss = criterion(output, target.long()) + 10 * torch.mean(torch.mul(w, w))

            if not math.isfinite(loss):
                logger.info('Loss is {}, stopping training'.format(loss))
                sys.exit(1)

            loss.backward()
            optimizer_vgg.step()
            optimizer_drama[k].step()

            metric_logger.meters['loss'].update(value=loss.item(),n=batch_size)



            if tb_writer is not None:
                tb_writer.add_scalar('train/loss_{}'.format(dic[k]), loss, n_iter)
            if profiler is not None:
                profiler.step()
                if n_iter >= n_iter_profiling:
                    break

    return n_iter

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)