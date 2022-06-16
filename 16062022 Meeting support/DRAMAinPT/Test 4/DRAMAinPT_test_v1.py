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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 24
img_w = 224
img_h = 224
test_iteration = 4000
torch.set_default_tensor_type(torch.DoubleTensor)

#prepare data
def return_path(path, keyword):
    for cur_path, dirs, files in os.walk(path):
        for file_name in files:
            if keyword in file_name:
                abs_path = os.path.join(os.path.abspath(cur_path),file_name)
                return abs_path

def preprocess(img, image_shape):
    img = skimage.transform.resize(img, image_shape)  # image_shape is [h,w]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    return img

def get(image_dir,csv_dir,batch_size,attribute):
    csv_reader = csv.reader(open(csv_dir), delimiter=';')
    data = []
    study = {'safe': '50a68a51fdc9f05596000002', 'lively': '50f62c41a84ea7c5fdd2e454',
             'beautiful': '5217c351ad93a7d3e7b07a64', 'wealthy': '50f62cb7a84ea7c5fdd2e458',
             'depressing': '50f62ccfa84ea7c5fdd2e459', 'boring': '50f62c68a84ea7c5fdd2e456'}
    study_id = study[attribute]
    for row in csv_reader:
        if (row[1]!='left' and row[1]!='right') or\
                (row[4]!= study_id) or \
                (return_path(image_dir,row[2]) == None) or \
                (return_path(image_dir, row[3]) == None):
            continue
        data.append({'dir':[return_path(image_dir,row[2]), return_path(image_dir,row[3])],\
                     'label':0 if row[1]=='left' else 1})
    num = len(data)
    print(attribute,'(number of comparison):',num)
    num_validation = int(num*0.05)//batch_size * batch_size
    num_test = int(num*0.35)//batch_size * batch_size
    num_train = num - num_test - num_validation
    data_train = data[0:num_train]
    data_validation = data[num_train:num_train+num_validation]
    data_test = data[num_train+num_validation:num]

    return data_train,data_validation,data_test

def get_files(image_dir,csv_dir,batch_size):#return list of images and labels
    return get(image_dir,csv_dir,batch_size,'beautiful'),\
            get(image_dir,csv_dir,batch_size,'boring'),\
            get(image_dir,csv_dir,batch_size,'depressing'),\
            get(image_dir,csv_dir,batch_size,'lively'),\
            get(image_dir,csv_dir,batch_size,'safe'),\
            get(image_dir,csv_dir,batch_size,'wealthy')

def get_selected_data(selected_data,image_width,image_height,batch_size):
    data_left = np.zeros((batch_size,3,image_height,image_width))
    data_right = np.zeros((batch_size,3,image_height,image_width))
    label = np.zeros((batch_size,1),dtype=int)
    assert len(selected_data) == batch_size
    for i,pair in enumerate(selected_data):
        image_left = io.imread(pair["dir"][0])
        image_right = io.imread(pair["dir"][1])#according to selected_data
        image_left = preprocess(image_left, [224, 224])
        image_right = preprocess(image_right, [224, 224])
        data_left[i] = image_left
        data_right[i] = image_right
        label[i] = np.array(pair["label"])
    return torch.from_numpy(data_left).to(device), torch.from_numpy(data_right).to(device), torch.from_numpy(label).to(device)#a list of tensor, a list of tensor, a list of tensor

def get_train(data,image_width, image_height, batch_size,iteration):
    if iteration >= len(data)//batch_size:
        iteration = iteration % (len(data)//batch_size)
    start = iteration*batch_size
    end = (iteration+1)*batch_size
    return get_selected_data(data[int(start):int(end)], image_width, image_height, batch_size)

def get_validation(data,start,end,image_width,image_height):
     selected_data = data[start:end]
     return get_selected_data(selected_data, image_width, image_height,end-start)


def get_test(data,start,end,image_width,image_height):
    selected_data = data[start:end]
    return get_selected_data(selected_data, image_width, image_height,end-start)

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

def shuffle(num, batch_size, iteration):
    for i in range(6):
        if iteration % (num[i] // batch_size) == 0:
            random.shuffle(data_train[i])

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

print('----------------------DRAMAinPT stars!--------------------------')
mynet= [MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device)]
vgg_extra = create_vgg19('./vgg_extractor')
vgg_extra.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_vgg = optim.SGD(vgg_extra.parameters(), lr=0.001, momentum=0.5)
optimizer_mynet = []
for i in range(6):
    optimizer_mynet.append(optim.Adam(mynet[i].parameters(), lr=0.001))

##prepare dataset
image_dir = './pp2/images'
csv_dir = './pp2/votes_convert.csv'
[train_beau,val_beau,test_beau],[train_bor,val_bor,test_bor],[train_depr,val_depr,test_depr],\
[train_lively,val_lively,test_lively],[train_safe,val_safe,test_safe],\
[train_wealth,val_wealth,test_wealth]= get_files(image_dir, csv_dir,batch_size)

data_train = [train_safe,train_lively,train_beau,train_wealth,train_depr,train_bor]
data_validation = [val_safe,val_lively,val_beau,val_wealth,val_depr,val_bor]
data_test = [test_safe,test_lively,test_beau,test_wealth,test_depr,test_bor]
num= []#order is:safe,lively,beau,wealthy,depr,bor
num_validation = []
num_test = []
for i in range(6):
    num.append(len(data_train[i]))
    num_validation.append(len(data_validation[i]))
    num_test.append(len(data_test[i]))
max_iterations = max(num)//batch_size
list_val_acc = [[],[],[],[],[],[]]
list_train_acc = [[],[],[],[],[],[]]
list_loss = [[],[],[],[],[],[]]
max_acc = [0,0,0,0,0,0]
acc_train = [0,0,0,0,0,0]
outputdir = './output'
t = datetime.date.today()
epoch = 10

#train loop
for epoch in range(epoch):
    sum_loss = [0,0,0,0,0,0]
    for iteration in range(max_iterations):
        shuffle(num, batch_size, iteration)
        for k in range(6):
            loss = 0.0
            optimizer_mynet[k].zero_grad()
            optimizer_vgg.zero_grad()
            img_left, img_right, label = get_train(data_train[k], img_w, img_h, batch_size, iteration)
            feature = fusion(img_left, img_right,vgg_extra)
            output,w = mynet[k](feature)
            label = label.squeeze()
            label = torch.tensor(label, dtype=torch.long)
            loss = criterion(output,label)+10*torch.mean(torch.mul(w, w))#why +10*nd.mean(w**2)
            acc_tmp = evaluate_accuracy(output, label)
            acc_train[k] += acc_tmp
            print('epoch=',epoch+1,'iteration=',iteration+1,'/',max_iterations,'loss',k,'=',loss)

            loss.backward()
            optimizer_vgg.step()
            optimizer_mynet[k].step()
            sum_loss[k]+=loss

            if iteration%test_iteration==test_iteration-1 or iteration== max_iterations-1:#after 4000 times or all ready train one epoch,start val
                if iteration%test_iteration==test_iteration-1:
                    list_train_acc[k].append(acc_train[k]/test_iteration)
                else:
                    list_train_acc[k].append(acc_train[k]/(max_iterations%test_iteration))

                print('**********test on data_validation of attribute',k,' start ************')
                acc_train[k] = 0
                times = num_validation[k]//batch_size
                accuracy = 0
                for i in range(times):
                    left_val,right_val,label = get_validation(data_validation[k],i*batch_size,(i+1)*batch_size,img_w,img_h)
                    val_feature = fusion(left_val,right_val,vgg_extra)
                    val_output,_ = mynet[k](val_feature)

                    acc_tmp = evaluate_accuracy(val_output, label)
                    accuracy += acc_tmp
                list_val_acc[k].append(accuracy/times)
                print('accuracy = ',accuracy/times)
                if accuracy/times>max_acc[k]:
                    max_acc[k] = accuracy/times
                    torch.save(vgg_extra.state_dict(), outputdir+'/'+'vgg_params_'+str(t))
                    torch.save(mynet[k].state_dict(), outputdir+'/'+'mynet'+str(k)+'_params_'+str(t))
                print('change of accuracy on data_train ' + str(k) + ':')
                for i in list_train_acc[k]:
                    print(i)
                print('change of accuracy on data_validation '+str(k)+':')
                for i in list_val_acc[k]:
                     print(i)
                print('**********test on data_validation of attribute',k,' finished ************')
                print('\n')
#print the results of six attributes
    print('\n')
    print('epoch '+str(epoch)+ 'finished, changes of losses as following: ')
    for j in range(6):
        if max_iterations == 0:
            continue
        list_loss[j].append(sum_loss[j]/max_iterations)
        print('change of loss'+str(j)+' every epoch:')
        for i in list_loss[j]:
            print(i)


#test
vgg_extra_test = models.vgg19(pretrained = False).to(device)
vgg_extra_test.load_state_dict(torch.load(outputdir+'/'+'vgg_params_'+str(t)))
vgg_extra_test.eval()
mynet_test= [MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device),MyNet().to(device)]
acc_test=[0,0,0,0,0,0]
for k in range(6):
    mynet_test[k].load_state_dict(torch.load(outputdir+'/'+'mynet'+str(k)+'_params_'+str(t)))
    mynet_test[k].eval()
    times = num_test[k]//batch_size
    print('\n')
    print('**********test on data_test of attribute',k,' start ************')
    for i in range(times):
        if i%20==0:
            print(i+1,'/',times)
        left_te,right_te,label = get_test(data_test[k],i*batch_size,(i+1)*batch_size,img_w,img_h)
        te_feature = fusion(left_te, right_te,vgg_extra_test)
        te_output,_ = mynet_test[k](te_feature)
        acc_tmp = evaluate_accuracy(te_output, label)
        acc_test[k] += acc_tmp
    print('accuracy on data_test of attribtue',k,':',acc_test[k]/times)

