import torch
import os
import numpy as np
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_mask_area(mask_dir, output_dir):
    mask_list = os.listdir(mask_dir)
    count = 0
    for mask_name in mask_list:
        print('name',mask_name)
        area = torch.zeros(65)
        mask_path = os.path.join(mask_dir, mask_name)
        cp = torch.load(mask_path)
        mask = cp['mask'].reshape(-1)
        print('mask',mask)
        for i in mask:
            if i !=0:
                area[i-1] +=1
        print('area',area)
        torch.save(area, os.path.join(output_dir, (mask_name)))
        count += 1
        print('process: ', count)

mask_dir = './output_info'
mask_output_n = './output_mask_area'
get_mask_area(mask_dir,mask_output_n)

