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
import torch.optim as optim
from pp2_dataset_arcnn import PP2HDF5Dataset
from torch.utils import tensorboard as tb
from model_arcnn import get_vgg_backbone,fusion_net,create_DRAMA,create_RSSNN,info_extractor
import utils



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_w = 224
img_h = 224
hdf5_original_path = './pp2_images.hdf5'
seg_path = './seg'
torch.set_default_tensor_type(torch.DoubleTensor)
t = datetime.date.today()


def get_args_parser():
    # atts, epoch, output_dir,resume
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='drama', help='architecture, drama or others')
    parser.add_argument('--img', '--img-folder', default='./pp2/images/', help='image folder or .hdf5 file')
    parser.add_argument('--dataset', default='./pp2/csv/',
                        help='dataset folder (with {train|val|test}.csv of 6 attributes')
    parser.add_argument('--atts', default=['safe','lively','beautiful','wealthy','depressing','boring'],
                        help='choose which attributes to feed the model')
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
    parser.add_argument('--lambda-ranking', type=float, default=1.,
                         help='regularization parameter in the loss for rscnn')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='regularization parameter in the total loss(vis loss and sem loss)')
    parser.add_argument('--t', type=float, default=0.5,
                        help='regularization parameter in the attention module for the fusion of pp2 img and seg mask')
    # parser.add_argument('--optim', default='sgd', help='optimizer (sgd or adam)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--log-freq', default=200, type=int, help='log progress every `log-freq` batches')
    parser.add_argument('--checkpoint-freq', default=1, type=int,
                        help='Save a checkpoint every `checkpoint-freq` epochs') 
    parser.add_argument('--output-dir', default='./output_arcnn_all', help='path where to save')
    parser.add_argument('--name', default='', help="Prefix name for the output dir (default: '')")
    parser.add_argument('--resume', default=None,
                        help='checkpoint path')
    parser.add_argument('--backbone-trainable-layers', default=3, type=int,
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

class RankingLoss(torch.nn.Module):
    def forward(self, left_score, right_score, target):
        """In the paper, target = 1 when left wins, -1 when right wins.
        Here, the opposite is implemented (see model.choice_to_numerical, so the loss function uses
        left_score - right_score.
        In the original paper, this loss uses the sum to aggregate. Here, we use the avg instead to have the same scale
        as the BCELoss (for SSCNN).
        """
        # map target from (0, 1) to (-1, 1). target needs to be in (0, 1) for the BCELoss
        target = target.clone()
        target[target == 0] = -1
        # TODO: check if a better loss can be used (e.g. by adding a margin in the max())
        return (torch.maximum(torch.tensor([0], device=target.device), 1 + target * (left_score - right_score)) ** 2).mean()


class RSSCNNLoss(torch.nn.Module):
    def __init__(self, l=1):
        """l: lambda"""
        super().__init__()
        self.l = l
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.ranking_loss = RankingLoss()

    def forward(self, model_output, target):
        """Return (loss, loss_details). loss is the main loss to train the model, loss_details is a dict with
        the value of each component of the loss.
        """
        sscnn_pred, rssccn_pred_left, rssccn_pred_right = model_output
        bce = self.bce_loss(sscnn_pred.squeeze(1).squeeze(1), target)
        ranking = self.ranking_loss(rssccn_pred_left.squeeze(1).squeeze(1), rssccn_pred_right.squeeze(1).squeeze(1), target)
        loss = bce + self.l * ranking
        return loss, {'loss_RSSNN+SSNN': loss, 'loss_SSNN': bce, 'loss_RSSNN': ranking}


class SSCNNLoss(torch.nn.Module):
    def __init__(self):
        """l: lambda"""
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, model_output, target):
        """Return (loss, loss_details). loss is the main loss to train the model, loss_details is a dict with
        the value of each component of the loss.
        """
        bce = self.loss(model_output.squeeze(1), target)
        return bce, {'BCE': bce}

class Data:
    def __init__(self, atts):
        self.data = dict(zip(atts,[[]] * len(atts)))
        
class Num:
    def __init__(self, atts):
        self.num = dict(zip(atts,[0] * len(atts)))

class Loader:
    def __init__(self, atts):
        self.loader = dict.fromkeys(atts)

def get(args):
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    data_train = Data(args.atts)
    data_validation = Data(args.atts)
    data_test = Data(args.atts)
    data_loader_train = Loader(args.atts)
    data_loader_test = Loader(args.atts)
    data_loader_val = Loader(args.atts)
    print('--------------------------star loading data-----------------------------------')
    for attribute in data_train.data:
        data_train.data[attribute] = PP2HDF5Dataset(votes_path=args.dataset+str(attribute) + '_train.csv',hdf5_path=hdf5_original_path, seg_path=seg_path)
        data_validation.data[attribute] = PP2HDF5Dataset(votes_path=args.dataset + str(attribute) + '_val.csv',hdf5_path=hdf5_original_path, seg_path=seg_path)
        data_test.data[attribute] = PP2HDF5Dataset(votes_path=args.dataset + str(attribute) + '_test.csv',hdf5_path=hdf5_original_path, seg_path=seg_path)


        train_sampler = torch.utils.data.RandomSampler(data_train.data[attribute])
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        val_sampler = torch.utils.data.SequentialSampler(data_validation.data[attribute])
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, args.test_batch_size, drop_last=True)

        test_sampler = torch.utils.data.SequentialSampler(data_test.data[attribute])
        test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, args.test_batch_size, drop_last=True)



        data_loader_train.loader[attribute] = torch.utils.data.DataLoader(data_train.data[attribute], batch_sampler=train_batch_sampler,
                                                                   num_workers=args.workers,
                                                                   pin_memory=True)
        data_loader_test.loader[attribute] = torch.utils.data.DataLoader(data_test.data[attribute], batch_sampler=test_batch_sampler,
                                                                  num_workers=args.workers,
                                                                  pin_memory=True)
        data_loader_val.loader[attribute] = torch.utils.data.DataLoader(data_validation.data[attribute],
                                                                 batch_sampler=val_batch_sampler,
                                                                 num_workers=args.workers,
                                                                 pin_memory=True)

    print('--------------------------finish loading data-----------------------------------')
    num_train = Num(args.atts)
    num_validation = Num(args.atts)
    num_test = Num(args.atts)
    for attribute in num_train.num:
        num_validation.num[attribute] = len(data_validation.data[attribute])
        num_test.num[attribute] = len(data_test.data[attribute])
        num_train.num[attribute] = len(data_train.data[attribute])
        print(attribute, '(train number of comparison):', num_train.num[attribute])
        print(attribute, '(val number of comparison):', num_validation.num[attribute])
        print(attribute, '(test number of comparison):', num_test.num[attribute])

    return data_loader_train.loader,data_loader_test.loader,data_loader_val.loader,num_train.num,num_validation.num,num_test.num


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


def attention_fuse(feature_ori,feature_seg,device,args):
    def f(z):
        z = z/1000
        f = (torch.exp(z) - torch.exp(-z)) / (torch.exp(z) + torch.exp(-z))
        return f

    A_r = torch.zeros(feature_ori.size(0), feature_ori.size(1),1,feature_ori.size(3)).to(device)
    for batch in range(feature_seg.size(0)):
        edist = torch.zeros(feature_ori.size(1), feature_ori.size(3)).to(device)
        dif = torch.zeros(feature_ori.size(1), feature_ori.size(3)).to(device)
        for h in range(feature_seg.size(2)):
            tmp_edist = torch.abs(torch.sub(feature_ori[batch,:,h,:], feature_seg[batch,:,h,:]))
            edist = torch.add(edist,tmp_edist)
            tmp_dif = torch.sub(feature_ori[batch,:,h,:], feature_seg[batch,:,h,:])
            dif = torch.add(dif,tmp_dif)
        A_r[batch,:,0,:] = f(torch.mul(edist, torch.sign(dif)))


    A_c = torch.zeros(feature_ori.size(0), feature_ori.size(1),feature_ori.size(2),1).to(device)
    for batch in range(feature_seg.size(0)):
        edist = torch.zeros(feature_ori.size(1), feature_ori.size(2)).to(device)
        dif = torch.zeros(feature_ori.size(1), feature_ori.size(2)).to(device)
        for w in range(feature_seg.size(3)):
            tmp_edist = torch.abs(torch.sub(feature_ori[batch,:,:,w], feature_seg[batch,:,:,w]))
            edist = torch.add(edist, tmp_edist)
            tmp_dif = torch.sub(feature_ori[batch,:,:,w], feature_seg[batch,:,:,w])
            dif = torch.add(dif, tmp_dif)
        A_c[batch,:,:,0] = f(torch.mul(edist, torch.sign(dif)))

    H = torch.zeros(feature_ori.size(0),feature_ori.size(1),feature_ori.size(2),feature_ori.size(3)).to(device)
    for batch in range(feature_seg.size(0)):
        for i in range(A_c.size(1)):
            H_s = torch.mul(torch.matmul(A_c[batch,i,:,:],A_r[batch,i,:,:]),  feature_seg)
            H[batch,i,:,:] = feature_ori[batch,i,:,:] + args.t * H_s[batch,i,:,:]
    return H



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

    #Setup device
    device = torch.device(args.device)

    logger.debug('Prepare data')
    data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test = get(args)#num is the number of training data
    max_iterations = max(num_train.values())//args.batch_size
    print('max_iterations',max_iterations)

    max_att = max(num_train, key=lambda k: num_train[k])
    atts_res = args.atts.copy()
    atts_res.remove(max_att)
    args.atts = [max_att]+atts_res
    print('train attributes order:', args.atts)
    att_num = len(args.atts)
    if att_num == 1:
        data_loader_train_final = zip(data_loader_train[args.atts[0]])
    elif att_num == 2:
        data_loader_train_final = zip(data_loader_train[args.atts[0]], cycle(data_loader_train[args.atts[1]]))
    elif att_num == 3:
        data_loader_train_final = zip(data_loader_train[args.atts[0]], cycle(data_loader_train[args.atts[1]]), cycle(data_loader_train[args.atts[2]]))
    elif att_num == 4:
        data_loader_train_final = zip(data_loader_train[args.atts[0]], cycle(data_loader_train[args.atts[1]]), cycle(data_loader_train[args.atts[2]]), cycle(data_loader_train[args.atts[3]]))
    elif att_num == 5:
        data_loader_train_final = zip(data_loader_train[args.atts[0]], cycle(data_loader_train[args.atts[1]]), cycle(data_loader_train[args.atts[2]]), cycle(data_loader_train[args.atts[3]]),
                                      cycle(data_loader_train[args.atts[4]]))
    elif att_num == 6:
        data_loader_train_final = zip(data_loader_train[args.atts[0]], cycle(data_loader_train[args.atts[1]]), cycle(data_loader_train[args.atts[2]]), cycle(data_loader_train[args.atts[3]]),
                                      cycle(data_loader_train[args.atts[4]]), cycle(data_loader_train[args.atts[5]]))
    else:
        print('The attributes number is wrong.')

    logger.debug('Creating model,optimizer and lr_scheduler')
    # extractor_vis: vgg for extracting original pp2 image feature
    # model_info: NN for seg annotation vector(task specific network list)
    # model_drama: the CNN task specific network list
    backbone_ori = get_vgg_backbone(19, args.backbone_trainable_layers)
    extractor_vis = fusion_net(backbone_ori)
    extractor_vis.to(device)

    extractor_sem = info_extractor().double()
    extractor_sem = extractor_sem.to(device)

    model_info = create_RSSNN(args.atts, device)
    model_drama = create_DRAMA(args.atts, device)


    
    optimizer_e_vis = torch.optim.SGD(extractor_vis.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    optimizer_e_sem = torch.optim.SGD(extractor_sem.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    optimizer_drama = {}
    lr_scheduler_drama = {}
    for att in args.atts:
        optimizer_drama[att] = optim.Adam(model_drama[att].parameters(), lr=args.lr)
        lr_scheduler_drama[att] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_drama[att], factor=0.1,
                                                                             min_lr=args.lr / 10000,
                                                                             patience=2)
    optimizer_info = {}
    lr_scheduler_info = {}
    for att in args.atts:
        optimizer_info[att] = optim.Adam(model_info[att].parameters(), lr=args.lr)
        lr_scheduler_info[att] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_info[att], factor=0.1,
                                                                            patience=2)
    lr_scheduler_e_vis = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_e_vis, factor=0.1, min_lr=args.lr / 10000,
                                                                  patience=2)
    lr_scheduler_e_sem = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_e_sem, factor=0.1, min_lr=args.lr / 10000,
                                                                    patience=2)

    criterion_vis = nn.CrossEntropyLoss()
    criterion_sem = RSSCNNLoss(args.lambda_ranking)

    n_iter = 0

    logger.info(
        'Trainable extractor_vis(vgg) parameters: {}'.format(
            sum(p.numel() for p in extractor_vis.parameters() if p.requires_grad)))
    logger.info(
        'Trainable extractor_sem(NN) parameters: {}'.format(
            sum(p.numel() for p in extractor_sem.parameters() if p.requires_grad)))

    for k in args.atts:
        logger.info('Trainable {} task specific parameters of model_drama: {}'.format(k, sum(
            p.numel() for p in model_drama[k].parameters() if p.requires_grad)))
    for k in args.atts:
        logger.info('Trainable {} task specific parameters of model_info: {}'.format(k, sum(
            p.numel() for p in model_info[k].parameters() if p.requires_grad)))


    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        cp = torch.load(args.resume)

        extractor_vis.load_state_dict(cp['extractor_vis'])
        extractor_sem.load_state_dict(cp['extractor_sem'])
        for k in args.atts:
            model_info[k].load_state_dict(cp['model_info_{}'.format(k)])
            model_drama[k].load_state_dict(cp['model_drama_{}'.format(k)])

        args.start_epoch = cp['epoch']
        n_iter = 0


    if args.test_only:
        for k in args.atts:
            header = 'Test:'+k
            test_loss, test_acc= evaluate(k,extractor_vis, extractor_sem, model_drama[k], model_info[k],
                                          criterion_vis, criterion_sem,
                                          data_loader_test[k], device, args.log_freq, header)
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
            train_one_epoch(max_iterations,
                            extractor_vis, extractor_sem, model_drama, model_info,
                            optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
                            lr_scheduler_e_vis, lr_scheduler_e_sem, lr_scheduler_drama, lr_scheduler_info,
                            criterion_vis, criterion_sem,
                            data_loader_train_final,  device, 0, 0, tb_writer, args.log_freq ,prof, n_iter_profiling)
    else:
        logger.info('Start training for {} epochs'.format(args.epochs))
        train(max_iterations,args, n_iter,
              criterion_vis,criterion_sem,
              data_loader_train_final, data_loader_test, data_loader_val,
              device, logger,
              extractor_vis, extractor_sem, model_drama, model_info,
              optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
              lr_scheduler_e_vis, lr_scheduler_e_sem, lr_scheduler_drama, lr_scheduler_info,
              tb_writer)




def train(max_iterations,args, n_iter,
          criterion_vis, criterion_sem,
          data_loader_train_final, data_loader_test, data_loader_val,
          device, logger,
          extractor_vis, extractor_sem, model_drama, model_info,
          optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
          lr_scheduler_e_vis, lr_scheduler_e_sem, lr_scheduler_drama, lr_scheduler_info,
          tb_writer):
    start_time = time.time()
    for epoch in range(args.start_epoch+1,args.start_epoch+1+args.epochs):
        n_iter += train_one_epoch(max_iterations,
                                  extractor_vis, extractor_sem, model_drama, model_info,
                                  optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
                                  lr_scheduler_e_vis, lr_scheduler_e_sem, lr_scheduler_drama, lr_scheduler_info,
                                  criterion_vis, criterion_sem,
                                  data_loader_train_final, device,epoch, n_iter, tb_writer,args.log_freq)

        #evaluation
        for k in args.atts:
            header = 'Val:'+k
            val_loss, val_accuracy = evaluate(k, extractor_vis, extractor_sem, model_drama[k], model_info[k],
                                              criterion_vis, criterion_sem, data_loader_val[k], device, args.log_freq, header)
            lr_scheduler_e_vis.step(val_loss)
            lr_scheduler_e_sem.step(val_loss)
            lr_scheduler_drama[k].step(val_loss)
            lr_scheduler_info[k].step(val_loss)

            if tb_writer is not None:
                tb_writer.add_scalar('lr_drama_{}'.format(k), optimizer_drama[k].param_groups[0]['lr'], n_iter)
                tb_writer.add_scalar('lr_info_{}'.format(k), optimizer_info[k].param_groups[0]['lr'], n_iter)
                tb_writer.add_scalar('val/loss_{}'.format(k), val_loss, n_iter)
                tb_writer.add_scalar('val/accuracy_{}'.format(k), val_accuracy, n_iter)

            if not args.no_test or epoch == (args.epochs - 1):
                header = 'Test:'+k
                test_loss, test_acc = evaluate(k, extractor_vis, extractor_sem, model_drama[k], model_info[k],
                                               criterion_vis, criterion_sem, data_loader_test[k], device, args.log_freq,header)
                if tb_writer is not None:
                    tb_writer.add_scalar('test/loss_{}'.format(k), test_loss, n_iter)
                    tb_writer.add_scalar('test/accuracy_{}'.format(k), test_acc, n_iter)

        tb_writer.add_scalar('lr_e_ori', optimizer_e_vis.param_groups[0]['lr'], n_iter)
        tb_writer.add_scalar('lr_e_info', optimizer_e_sem.param_groups[0]['lr'], n_iter)

        if ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == (args.epochs - 1):
            cp = {
                'extractor_vis': extractor_vis.state_dict(),
                'extractor_sem': extractor_sem.state_dict(),
                'optimizer_e_vis': optimizer_e_vis.state_dict(),
                'optimizer_e_sem': optimizer_e_sem.state_dict(),
                'lr_scheduler_e_vis': lr_scheduler_e_vis.state_dict(),
                'lr_scheduler_e_sem': lr_scheduler_e_sem.state_dict(),
                'args': args,
                'epoch': epoch,
                'n_iter': n_iter
            }
            for k in args.atts:
                cp['model_drama_{}'.format(k)] = model_drama[k].state_dict()
                cp['optimizer_drama_{}'.format(k)] = optimizer_drama[k].state_dict()
                cp['lr_scheduler_drama_{}'.format(k)] = lr_scheduler_drama[k].state_dict()
                cp['model_info_{}'.format(k)] = model_info[k].state_dict()
                cp['optimizer_info_{}'.format(k)] = optimizer_info[k].state_dict()
                cp['lr_scheduler_info_{}'.format(k)] = lr_scheduler_info[k].state_dict()

            torch.save(cp, os.path.join(args.output_dir, 'final_model_{}.pt'.format(epoch)))
            logger.info('[Epoch {}] Checkpoint saved'.format(epoch))

    logger.info('Training time (total) {}'.format(time_elapsed(start_time)))

@torch.no_grad()
def evaluate(k, extractor_vis, extractor_sem, model_drama, model_info,
             criterion_vis, criterion_sem, data_loader, device, log_freq, header):

    # TODO: evaluate with batch_size > 1 for faster computation
    """Return the mean loss and accuracy of the model on the dataset"""
    logger = logging.getLogger('evaluator')
    extractor_vis.eval()
    extractor_sem.eval()
    model_info.eval()
    model_drama.eval()
    metric_logger = utils.MetricLogger(logger=logger)
    total_vis_loss, total_sem_loss, n_samples = 0,0,0

    for img1, img2, mask1, mask2, info1, info2, target in metric_logger.log_every(data_loader, log_freq, header):
        img1, img2, mask1, mask2, info1, info2, target = img1.to(device), img2.to(device), mask1.to(device), \
                                                         mask2.to(device), info1.to(device), info2.to(device),target.to(device)
        batch_size = img1.size(0)
        n_samples += batch_size

        # visual stream
        feature_mask = extractor_vis(mask1, mask2,batch_size)
        feature_ori = extractor_vis(img1, img2,batch_size)
        feature = attention_fuse(feature_ori,feature_mask,device,args)
        output_vis, w = model_drama(feature)
        loss_vis = criterion_vis(output_vis, target.long()) + 10 * torch.mean(torch.mul(w,w))
        total_vis_loss += loss_vis.item()*batch_size
        acc_tmp = evaluate_accuracy(output_vis, target)
        metric_logger.meters['acc_vis_{}'.format(k)].update(acc_tmp, n=batch_size)
        metric_logger.meters['loss_vis_{}'.format(k)].update(loss_vis.item(), n=batch_size)

        #semantic stream
        feature_info1 = extractor_sem(info1)
        feature_info2 = extractor_sem(info2)
        output_sem = model_info(feature_info1, feature_info2)
        loss_sem, loss_details_sem = criterion_sem(output_sem, target)
        total_sem_loss += loss_sem.item() * batch_size

        ssnn_pred, rssn_pred_left, rssn_pred_right = output_sem
        ssnn_acc = (torch.abs(target - torch.sigmoid(ssnn_pred.squeeze())) <= 0.5).sum().item() / batch_size
        rssnn_pred = (rssn_pred_right >= rssn_pred_left).squeeze() * torch.tensor(1)
        rssnn_acc = (torch.abs(target - rssnn_pred) <= 0.5).sum().item() / batch_size
        metric_logger.meters['acc_rssnn_{}'.format(k)].update(rssnn_acc, n=batch_size)
        metric_logger.meters['acc_ssnn_{}'.format(k)].update(ssnn_acc, n=batch_size)
        for name, l in loss_details_sem.items():
            metric_logger.meters['{}_{}'.format(name,k)].update(l.item(), n=batch_size)

        #sum losses
        loss = (1-args.beta)*loss_vis + args.beta * loss_sem
        metric_logger.meters['loss_{}'.format(k)].update(loss.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    logger.info(header + ' ' + str(metric_logger))

    vis_loss_av = total_vis_loss / n_samples
    sem_loss_av = total_sem_loss / n_samples
    loss_av = (1-args.beta)*loss_vis + args.beta * loss_sem

    return loss_av, metric_logger.meters['acc_vis_{}'.format(k)].global_avg


def train_one_attribute(k,d,device,
                        extractor_vis,extractor_sem,model_drama,model_info,
                        optimizer_e_vis,optimizer_e_sem,optimizer_drama, optimizer_info,
                        criterion_vis, criterion_sem,logger,metric_logger,tb_writer,profiler,n_iter):
    img1, img2, mask1, mask2, info1, info2, target = d
    img1, img2, mask1, mask2, info1, info2, target = img1.to(device), img2.to(device), \
                                                     mask1.to(device), mask2.to(device), \
                                                     info1.to(device), info2.to(device), \
                                                     target.to(device)

    batch_size = img1.size(0)

    optimizer_e_vis.zero_grad()
    optimizer_e_sem.zero_grad()
    optimizer_drama.zero_grad()
    optimizer_info.zero_grad()

    # visual stream
    feature_ori = extractor_vis(img1, img2, batch_size)
    feature_mask = extractor_vis(mask1, mask2, batch_size)
    feature = attention_fuse(feature_ori, feature_mask,device,args)
    output_vis, w_vis = model_drama(feature)
    loss_vis = criterion_vis(output_vis, target.long()) + 10 * torch.mean(torch.mul(w_vis, w_vis))

    # semantic stream
    feature_info1 = extractor_sem(info1)
    feature_info2 = extractor_sem(info2)
    output_sem = model_info(feature_info1, feature_info2)
    loss_sem, loss_detail_sem = criterion_sem(output_sem, target)

    #sum losses
    loss = (1-args.beta)*loss_vis + args.beta * loss_sem


    if not math.isfinite(loss):
        logger.info('Loss is {}, stopping training'.format(loss))
        sys.exit(1)

    loss.backward()
    optimizer_e_vis.step()
    optimizer_e_sem.step()
    optimizer_drama.step()
    optimizer_info.step()

    metric_logger.meters['loss_vis_{}'.format(k)].update(value=loss_vis.item(), n=batch_size)
    metric_logger.meters['loss_sem_{}'.format(k)].update(value=loss_sem.item(), n=batch_size)
    metric_logger.meters['loss_{}'.format(k)].update(value=loss.item(), n=batch_size)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_{}'.format(k), loss, n_iter)
    if profiler is not None:
        profiler.step()



def train_one_iteration(epoch,n_iter,data,device,
                        optimizer_e_vis,optimizer_e_sem,optimizer_drama,optimizer_info,
                        lr_scheduler_e_vis,lr_scheduler_e_sem,lr_scheduler_drama,lr_scheduler_info,
                        extractor_vis,extractor_sem,model_drama,model_info,
                        criterion_vis,criterion_sem,
                        logger, metric_logger,tb_writer,n_iter_profiling,profiler):
    n_iter += 1
    for i, d in enumerate(data):
        train_one_attribute(args.atts[i],d,device,
                            extractor_vis, extractor_sem,model_drama[args.atts[i]],model_info[args.atts[i]],
                            optimizer_e_vis,optimizer_e_sem,optimizer_drama[args.atts[i]],optimizer_info[args.atts[i]],
                            criterion_vis, criterion_sem,
                            logger,metric_logger,tb_writer,profiler,n_iter)
        if profiler is not None:
            if n_iter >= n_iter_profiling:
                break

    if n_iter%1000 == 0:
        cp = {
            'extractor_vis': extractor_vis.state_dict(),
            'extractor_sem': extractor_sem.state_dict(),
            'optimizer_e_vis': optimizer_e_vis.state_dict(),
            'optimizer_e_sem': optimizer_e_sem.state_dict(),
            'lr_scheduler_e_vis': lr_scheduler_e_vis.state_dict(),
            'lr_scheduler_e_sem': lr_scheduler_e_sem.state_dict(),
            'args': args,
            'epoch': epoch,
            'n_iter': n_iter
        }
        for k in args.atts:
            cp['model_drama_{}'.format(k)] = model_drama[k].state_dict()
            cp['optimizer_drama_{}'.format(k)] = optimizer_drama[k].state_dict()
            cp['lr_scheduler_drama_{}'.format(k)] = lr_scheduler_drama[k].state_dict()
            cp['model_info_{}'.format(k)] = model_info[k].state_dict()
            cp['optimizer_info_{}'.format(k)] = optimizer_info[k].state_dict()
            cp['lr_scheduler_info_{}'.format(k)] = lr_scheduler_info[k].state_dict()

        torch.save(cp, os.path.join(args.output_dir, 'model_{}_{}.pt'.format(epoch,n_iter)))
        logger.info('[Epoch {}_{}] Checkpoint saved'.format(epoch,n_iter))

    return n_iter


def train_one_epoch(max_iterations,
                    extractor_vis,extractor_sem,model_drama,model_info,
                    optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
                    lr_scheduler_e_vis,lr_scheduler_e_sem,lr_scheduler_drama,lr_scheduler_info,
                    criterion_vis,criterion_sem,
                    data_loader_train_final, device, epoch,
                    n_iter, tb_writer, log_freq, profiler=None,n_iter_profiling=None):
    logger = logging.getLogger('trainer')

    extractor_vis.train()
    extractor_sem.train()
    for model in model_drama.values():
        model.train()
    for model in model_info.values():
        model.train()

    metric_logger = utils.MetricLogger(logger=logger)
    header = 'Epoch: [{}]'.format(epoch)
    for data in metric_logger.log_every_train(max_iterations,data_loader_train_final, log_freq, header):
        n_iter = train_one_iteration(epoch,n_iter, data, device,
                                     optimizer_e_vis, optimizer_e_sem, optimizer_drama, optimizer_info,
                                     lr_scheduler_e_vis, lr_scheduler_e_sem, lr_scheduler_drama, lr_scheduler_info,
                                     extractor_vis, extractor_sem, model_drama, model_info,
                                     criterion_vis, criterion_sem,
                                     logger, metric_logger, tb_writer, n_iter_profiling,profiler)
    return n_iter

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
