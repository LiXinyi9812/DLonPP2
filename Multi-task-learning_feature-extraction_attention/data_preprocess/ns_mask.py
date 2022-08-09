import torch
import os
import numpy as np
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_pixel_mean(mask_dir,sample):
    R_sum = 0
    count = 0

    for mask_name in sample:
        mask_path = os.path.join(mask_dir, mask_name)
        print(mask_path)
        print(os.path.isdir(mask_path))
        cp = torch.load(mask_path)
        mask = (cp['mask']/64).float()
        R_sum += mask[:, :].mean()
        count += 1
        print('mean',count)
    R_mean = R_sum / count
    print('R_mean:{}'.format(R_mean))
    return R_mean


def get_image_pixel_std(mask_dir, sample,mask_mean):
    R_squared_mean = 0
    count = 0
    for mask_name in sample:
        mask_path = os.path.join(mask_dir, mask_name)
        cp = torch.load(mask_path)
        mask = (cp['mask']/64).float()
        mask = mask - mask_mean
        R_squared_mean += torch.sqrt(torch.mean(torch.flatten(torch.square(mask[:, :]))))
        count += 1
        print('std',count)
    R_std = math.sqrt(R_squared_mean / count)
    print('R_std:{}'.format(R_std))
    return R_std

def normalizer_mask(mask_dir, output_dir,mask_mean, mask_std):
    mask_list = os.listdir(mask_dir)
    count = 0
    for mask_name in mask_list:
        mask_path = os.path.join(mask_dir, mask_name)
        cp = torch.load(mask_path)
        mask = (cp['mask']/64).float()
        mask = (mask - mask_mean)/mask_std
        mask = mask.repeat(3,1,1)
        cp['mask'] = mask.double()
        torch.save(cp, os.path.join(output_dir, (mask_name)))
        count += 1
        print('nor', count)

mask_dir = './output_info'
mask_output_n = './output_info_n'

mask_list = os.listdir(mask_dir)
filenumber = len(mask_list)
rate = 0.01
print(int(rate*filenumber))
sample = random.sample(mask_list,int(rate*filenumber))
#sample = mask_list[:5000]
print(sample)
mean = get_image_pixel_mean(mask_dir,sample)
std = get_image_pixel_std(mask_dir,sample,mean)
normalizer_mask(mask_dir,mask_output_n,mean,std)

