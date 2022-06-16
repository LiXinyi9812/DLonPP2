import numpy as np
import csv
import random
from mxnet import image
from mxnet import nd
import os

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def return_path(path, keyword):
    for cur_path, dirs, files in os.walk(path):
        for file_name in files:
            if keyword in file_name:
                abs_path = os.path.join(os.path.abspath(cur_path),file_name)
                return abs_path

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)

def get(image_dir,csv_dir,batch_size,attribute):
    csv_reader = csv.reader(open(csv_dir), delimiter=';')
    data = []
    study = {'safe': '50a68a51fdc9f05596000002', 'lively': '50f62c41a84ea7c5fdd2e454',
             'beautiful': '5217c351ad93a7d3e7b07a64', 'wealthy': '50f62cb7a84ea7c5fdd2e458',
             'depressing': '50f62ccfa84ea7c5fdd2e459', 'boring': '50f62c68a84ea7c5fdd2e456'}
    study_id = study[attribute]
    for row in csv_reader:
        if (row[1] != 'left' and row[1] != 'right') or \
                (row[4] != study_id) or \
                (return_path(image_dir, row[2]) == None) or \
                (return_path(image_dir, row[3]) == None):
            continue
        # data.append({"dir":[image_dir+'/'+row[2]+'.jpg',image_dir+'/'+row[3]+'.jpg'],\
        #             "label":0 if row[1]=='left' else 1})
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

def get_selected_data(selected_data,image_width,image_height,batch_size,ctx):
    data_left = nd.zeros([batch_size,3,image_height,image_width],ctx=ctx)
    data_right = nd.zeros([batch_size,3,image_height,image_width],ctx=ctx)
    label = np.zeros([batch_size,1],np.int)
    assert len(selected_data)==batch_size
    for i,pair in enumerate(selected_data):
        image_left = image.imread(pair["dir"][0])
        image_right = image.imread(pair["dir"][1])
        image_left = preprocess(image_left, (224, 224))
        image_right = preprocess(image_right, (224, 224))
        data_left[i] = image_left[0]
        data_right[i] = image_right[0]
        label[i] = pair["label"]
    return data_left,data_right,label
def get_batch(data,image_width, image_height, batch_size,ctx,iteration):
    if iteration>=len(data)//batch_size:
        iteration = iteration%(len(data)//batch_size)
    start = iteration*batch_size
    end = (iteration+1)*batch_size
    return get_selected_data(data[int(start):int(end)], image_width, image_height, batch_size,ctx)
def get_validation(data,start,end,image_width,image_height,ctx):
    selected_data = data[start:end]
    return get_selected_data(selected_data, image_width, image_height,end-start,ctx)
def get_test(data,start,end,image_width,image_height,ctx):
    selected_data = data[start:end]
    return get_selected_data(selected_data, image_width, image_height,end-start,ctx)
