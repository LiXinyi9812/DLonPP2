# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import mxnet as mx
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
from mxnet import image
from mxnet.gluon import nn
import csv
import random
from mxnet import initializer
from mxnet import autograd
import numpy as np
import input_data

ctx = mx.gpu(0)
batch_size = 48
img_w = 224
img_h = 224
test_iteration = 4000


def create_vgg(model_path):
    vgg = models.vgg19(pretrained = False)
    vgg.collect_params().load(filename=model_path)
    return vgg

def get_vgg(vgg):
    net = nn.Sequential()
    for i in range(37):
        net.add(vgg.features[i])
    return net

class MyNet(nn.Block):
    def __init__(self, **kwargs):
        super(MyNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels= 10,kernel_size=3,activation='softrelu')
            self.conv2 = nn.Conv2D(channels= 10,kernel_size=3,activation='softrelu')
            self.conv3 = nn.Conv2D(channels= 10,kernel_size=3,activation='softrelu')
            self.dense = nn.Dense(2)
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        res = self.dense(x_3)#no use of activation function
        return  res,self.dense.weight.data()

vgg = create_vgg('VGG_extractor_DRAMA')
mynet= [MyNet(),MyNet(),MyNet(),MyNet(),MyNet(),MyNet()]
print('finish loading networks')
for i in range(6):
    mynet[i].initialize(ctx = ctx)
vgg_extra = get_vgg(vgg)
vgg_extra.collect_params().reset_ctx(ctx)
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
trainer_vgg = mx.gluon.Trainer(vgg_extra.collect_params(),'sgd',{'learning_rate':0.001,'momentum':0.5}) 
trainer = []
for i in range(6):
    trainer.append(mx.gluon.Trainer(mynet[i].collect_params(), 'adam',{'learning_rate':0.001})) 
t = '03152330'#timestamp
image_dir = './images'
csv_dir = './votes_convert.csv'
[train_beau,val_beau,test_beau],[train_bor,val_bor,test_bor],[train_depr,val_depr,test_depr],\
[train_lively,val_lively,test_lively],[train_safe,val_safe,test_safe],\
[train_wealth,val_wealth,test_wealth]= input_data.get_files(image_dir, csv_dir,batch_size)
data_train = [train_safe,train_lively,train_beau,train_wealth,train_depr,train_bor]
data_validation = [val_safe,val_lively,val_beau,val_wealth,val_depr,val_bor]
data_test = [test_safe,test_lively,test_beau,test_wealth,test_depr,test_bor]
print('finish loading data')
num= []
num_validation = []
num_test = []
for i in range(6):
    num.append(len(data_train[i]))
    num_validation.append(len(data_validation[i]))
    num_test.append(len(data_test[i]))

print('num is: ', num)
max_iterations = num[0]//batch_size
list_val_acc = [[],[],[],[],[],[]]
list_train_acc = [[],[],[],[],[],[]]
list_loss = [[],[],[],[],[],[]]
max_acc = [0,0,0,0,0,0]
acc_train = [0,0,0,0,0,0]
#train loop
for epoch in range(10):
    sum_loss = [0,0,0,0,0,0]
    for iteration in range(max_iterations):
        for i in range(6):
            if iteration%(num[i]//batch_size)==0:
                random.shuffle(data_train[i])
        for k in range(6):
            img_left,img_right,label = input_data.get_batch(data_train[k],img_w,img_h, batch_size,ctx,iteration)
            with autograd.record():
                left_feature = vgg_extra(img_left)
                right_feature = vgg_extra(img_right)
                left_feature = left_feature.reshape((batch_size,-1,49))
                right_feature = right_feature.reshape((batch_size,-1,49))
                left_feature = nd.transpose(left_feature,axes=(0,2,1))
                join_feature = nd.batch_dot(left_feature,right_feature)
                feature = join_feature.expand_dims(axis=1)
                output,w = mynet[k](feature)
                label = nd.array(label).copyto(ctx)
                loss = nd.mean(softmax_cross_entropy(output,label))+10*nd.mean(w**2)
                acc = mx.metric.Accuracy()
                acc.update(preds = [output],labels = [label])
                acc_train[k]+= acc.get()[1]
                if iteration%10==0:
                    print(output)
                print('epoch=',epoch+1,'iteration=',iteration,'/',max_iterations,'loss',k,'=',nd.mean(loss).asscalar())
            loss.backward()
            trainer_vgg.step(1)
            trainer[k].step(1)
            sum_loss[k]+=nd.mean(loss)
            if iteration%test_iteration==test_iteration-1 or iteration== max_iterations-1:
                if iteration%test_iteration==test_iteration-1:
                    list_train_acc[k].append(acc_train[k]/test_iteration)
                else:
                    list_train_acc[k].append(acc_train[k]/(max_iterations%test_iteration))
                acc_train[k] = 0
                times = num_validation[k]//batch_size
                accuracy = 0
                print('**********test on validation',k,' start ************')
                if k == 1:
                    print(data_validation[k])
                for i in range(times):
                    #left_val,right_val,label = input_data.get_validation(data_validation[k],i*batch_size,(i+1)*batch_size,img_w,img_h,ctx)
                    left_val, right_val, label = input_data.get_validation(data_validation[k],img_w, img_h, batch_size,ctx,iteration)
                    val_left_fea = vgg_extra(left_val)
                    val_right_fea = vgg_extra(right_val)
                    val_label = nd.array(label).copyto(ctx)
                    val_left_fea = val_left_fea.reshape((batch_size,-1,49))
                    val_right_fea = val_right_fea.reshape((batch_size,-1,49))
                    val_left_fea = nd.transpose(val_left_fea,axes=(0,2,1))
                    val_joint_fea = nd.batch_dot(val_left_fea,val_right_fea)
                    val_feature = val_joint_fea.expand_dims(axis=1)
                    val_output,_ = mynet[k](val_feature)
                    acc = mx.metric.Accuracy()
                    acc.update(preds = [val_output],labels = [val_label])
                    accuracy += acc.get()[1]
                list_val_acc[k].append(accuracy/times)
                print('**********test on data_validation',k,' finished,accuracy = ',accuracy/times,' **********')
                if accuracy/times>max_acc[k]:
                    max_acc[k] = accuracy/times
                    vgg_extra.collect_params().save(filename='vgg_params'+str(k)+'_'+t+'.params')
                    mynet[k].collect_params().save(filename='mynet_params'+str(k)+'_'+t+'.params')
                print('chage of accuracy on data_validation'+str(k)+':')
                for i in list_val_acc[k]:
                    print(i)
                print('chage of accuracy on train'+str(k)+':')
                for i in list_train_acc[k]:
                    print(i)
    for j in range(6):
        list_loss[j].append(sum_loss[j]/max_iterations)
        print('chage of loss'+str(j)+' every epoch:')
        for i in list_loss[j]:
            print(i)
for k in range(6):
    vgg_extra.collect_params().load(filename='vgg_params'+str(k)+'_'+t+'.params',ctx=ctx)
    mynet[k].collect_params().load(filename='mynet_params'+str(k)+'_'+t+'.params',ctx=ctx)
    times = num_test[k]//batch_size
    accuracy = 0
    print('**********test on data_test',k,' start ************')
    for i in range(times):
        if i%20==0:
            print(i,'/',times)
        left_val,right_val,label = input_data.get_test(data_test[k],i*batch_size,(i+1)*batch_size,img_w,img_h,ctx)
        val_left_fea = vgg_extra(left_val)
        val_right_fea = vgg_extra(right_val)
        val_label = nd.array(label).copyto(ctx)
        val_left_fea = val_left_fea.reshape((batch_size,-1,49))
        val_right_fea = val_right_fea.reshape((batch_size,-1,49))
        val_left_fea = nd.transpose(val_left_fea,axes=(0,2,1))
        val_joint_fea = nd.batch_dot(val_left_fea,val_right_fea)
        val_feature = val_joint_fea.expand_dims(axis=1)
        val_output,_ = mynet[k](val_feature)
        acc = mx.metric.Accuracy()
        acc.update(preds = [val_output],labels = [val_label])
        accuracy += acc.get()[1]
    print('accuracy on data_test:',accuracy/times)
