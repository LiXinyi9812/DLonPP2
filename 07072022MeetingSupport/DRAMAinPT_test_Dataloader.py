#!/usr/bin/env python3

import datetime
import numpy as np
import csv
import random
import torch
import skimage.io as io
import skimage
import os
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
from memory_profiler import profile


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 48
img_w = 224
img_h = 224
test_iteration = 4000
torch.set_default_tensor_type(torch.DoubleTensor)

image_dir = './pp2/images/'
csv_dir = './pp2/csv/'
outputdir = './output'
hdf5_path = './pp2_images.hdf5'
t = datetime.date.today()
epoch = 2
workers = 4

#for test: csv_dir,workers,epoch,batchsize


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


def fusion(img_l,img_r,vgg_extra):
    left_feature = vgg_extra.features(img_l)
    left_feature = vgg_extra.avgpool(left_feature)
    right_feature = vgg_extra.features(img_r)
    right_feature = vgg_extra.avgpool(right_feature)
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
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
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

def create_vgg19(model_path):
    model = models.vgg19(pretrained = False)
    model.load_state_dict(torch.load(model_path))
    return model

def plot_loss_accuracy(list_loss,list_train_acc,list_val_acc,epoch):
    for k in list_loss:
        x_l = np.arange(0, epoch, 1)
        y_l = np.array(torch.tensor(list_loss[k],device='cpu'))
        print(x_l,y_l)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#preprocess data
# renameIMG.rename(image_dir)
# csv_split.splitAtt(csv_path,csv_dir)
#
# test_rate = 0.35
# val_rate =  0.05/0.6
# for att in ['safe','lively','beautiful','wealthy','depressing','boring']:
#     csv_split.splitTVT(csv_dir+str(att)+'.csv', att, test_rate, val_rate,csv_dir,batch_size)


#prepare network
print('DRAMAinPT_final starts!')
#mynet = {'safe':MyNet().to(device),'lively':MyNet().to(device),'beautiful':MyNet().to(device),'wealthy':MyNet().to(device),'depressing':MyNet().to(device),'boring':MyNet().to(device)}
mynet = [MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device)]
vgg_extra = create_vgg19('./vgg_extractor')
print('finish loading networks')
vgg_extra.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_vgg = optim.SGD(vgg_extra.parameters(), lr=0.001, momentum=0.5)
optimizer_mynet = []
num_par = 0
for net in mynet:
    optimizer_mynet.append(optim.Adam(net.parameters(), lr=0.001))
    num_par += count_parameters(net)
num_par += count_parameters(vgg_extra)
print('numbers of DRAMA parameters: ', num_par)

#prepare data
data_loader_train,data_loader_test,data_loader_val,num_train,num_validation,num_test = get(csv_dir)#num is the number of training data
list_val_acc = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
list_train_acc = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
list_loss = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}
max_acc = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
acc_train = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
max_iterations = max(num_train.values())//batch_size

data_loader_train_final = zip(data_loader_train['safe'], \
                              cycle(data_loader_train['lively']), \
                              cycle(data_loader_train['beautiful']), \
                              cycle(data_loader_train['wealthy']), \
                              cycle(data_loader_train['depressing']), \
                              cycle(data_loader_train['boring']))



dic={0:'safe',1:'lively',2:'beautiful',3:'wealthy',4:'depressing',5:'boring'}
#train loop


for epo in range(epoch):
    sum_loss = {'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}
    for iteration, img1img2target in enumerate(data_loader_train_final):
        print('batch: ', iteration,'starts!')
        for k in range(6):
            img1, img2, target = img1img2target[k]
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            optimizer_mynet[k].zero_grad()
            optimizer_vgg.zero_grad()
            feature = fusion(img1, img2,vgg_extra)
            output,w = mynet[k](feature)
            loss = criterion(output,target.long())+10*torch.mean(torch.mul(w, w))#why +10*nd.mean(w**2)
            acc_tmp = evaluate_accuracy(output, target)
            acc_train[dic[k]] += acc_tmp
            print('epoch=', epo + 1, 'iteration=', iteration + 1, '/', max_iterations, 'loss of ', dic[k], '=', loss)

            loss.backward()
            optimizer_vgg.step()
            optimizer_mynet[k].step()
            sum_loss[dic[k]]+=loss



            #summary accuracy
            if iteration%test_iteration==test_iteration-1 or iteration== max_iterations-1:#after 4000 times or all ready train one epoch,start val
                if iteration%test_iteration==test_iteration-1:
                    list_train_acc[dic[k]].append(acc_train[dic[k]]/test_iteration)
                else:
                    list_train_acc[dic[k]].append(acc_train[dic[k]]/(max_iterations%test_iteration))

                #validation
                print('**********test on data_validation of attribute',dic[k],' start ************')
                acc_train[dic[k]] = 0
                accuracy_val = 0
               #for times, img1img2target in enumerate(data_loader_val_final):
                for times, (img1_val, img2_val, target_val) in enumerate(data_loader_val[dic[k]]):
                    img1_val, img2_val, target_val = img1_val.to(device), img2_val.to(device), target_val.to(device)
                    print(dic[k],target_val)
                    val_feature = fusion(img1_val,img2_val,vgg_extra)
                    val_output,_ = mynet[k](val_feature)
                    print(dic[k], val_output)

                    acc_tmp_val = evaluate_accuracy(val_output, target_val)
                    accuracy_val += acc_tmp_val
                    print('times= ',(times+1))
                list_val_acc[dic[k]].append(accuracy_val/(times+1))
                print('validation accuracy of ',dic[k], '=', accuracy_val/(times+1))
                if accuracy_val/(times+1)>max_acc[dic[k]]:
                    max_acc[dic[k]] = accuracy_val/(times+1)
                    torch.save(vgg_extra.state_dict(), outputdir+'/'+'vgg_params')
                    torch.save(mynet[k].state_dict(), outputdir+'/'+'mynet'+dic[k]+'_params')
                #show changes
                print('change of accuracy on data_train ' + str(dic[k]) + ':')
                for i in list_train_acc[dic[k]]:
                    print(i)
                print('change of accuracy on data_validation '+str(dic[k])+':')
                for i in list_val_acc[dic[k]]:
                     print(i)
                print('**********test on data_validation of attribute',dic[k],' finished ************')
                print('\n')
        print('batch: ', iteration, 'finished!')

    print('\n')
    print('epoch '+str(epo)+ ' finished, changes of losses as following: ')
    for j in sum_loss:
        if max_iterations == 0:
            continue
        list_loss[j].append(sum_loss[j]/max_iterations)
        print('change of loss '+str(j)+' every epoch:')
        for i in list_loss[j]:
            print(i)


#test
vgg_extra_test = models.vgg19(pretrained = False).to(device)
vgg_extra_test.load_state_dict(torch.load(outputdir+'/'+'vgg_params'))
vgg_extra_test.eval()
mynet_test = [MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device)]
#mynet_test= {'safe':MyNet().to(device),'lively':MyNet().to(device),'beautiful':MyNet().to(device),'wealthy':MyNet().to(device),'depressing':MyNet().to(device),'boring':MyNet().to(device)}
acc_test={'safe':0,'lively':0,'beautiful':0,'wealthy':0,'depressing':0,'boring':0}

for k in range(6):
    print('\n')
    print('**********test on data_test of attribute ', dic[k], ' start ************')
    times = 0
    mynet_test[k].load_state_dict(torch.load(outputdir + '/' + 'mynet' + dic[k] + '_params'))
    mynet_test[k].eval()
    for iteration, (img1, img2, target) in enumerate(data_loader_test[dic[k]]):
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        print(target, dic[k])
        te_feature = fusion(img1, img2,vgg_extra_test)
        te_output,_ = mynet_test[k](te_feature)
        acc_tmp = evaluate_accuracy(te_output, target)
        acc_test[dic[k]] += acc_tmp
        times = times+1
        print('accuracy on data_test of attribtue',dic[k],':',acc_test[dic[k]]/times)

#plot
print(epoch)
print(np.arange(epoch))
plot_loss_accuracy(list_loss,list_train_acc,list_val_acc,epoch)
