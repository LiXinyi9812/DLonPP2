#!/usr/bin/env python3

import argparse
import time
import numpy as np
import math
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
import pandas as pd
import joblib
import time




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_w = 224
img_h = 224
hdf5_original_path = './pp2_images.hdf5'
seg_path = './seg'
votes_path = './pp2/votes'
torch.set_default_tensor_type(torch.DoubleTensor)
t = datetime.date.today()


def get_args_parser():
    # atts, epoch, output_dir,resume
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset', default='./pp2/test_csv/',
                        help='dataset folder (with {train|val|test}.csv of 6 attributes')
    parser.add_argument('--atts', default=['safe','lively','beautiful','wealthy','boring','depressing'],
                        help='choose which attributes to feed the model')
    parser.add_argument('--output-dir', default='./output_rf/', help='path where to save')
    parser.add_argument('--resume', default=None,
                        help='checkpoint path')
    return parser


def get_seg(filename):
    cp = torch.load(os.path.join(seg_path,filename+'.pt'))
    info = cp['info']
    return info

def choice_to_numerical(choice):
    if choice == 'left':
        return 0.
    elif choice == 'right':
        return 1.
    raise ValueError('choice must be either left or right, no other value is supported')

def get_dataset(votes_path):
    data = []
    target = []
    # feature_names = ['None', 'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane',
    #                  'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road',
    #                  'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
    #                  'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain', 'Sand', 'Sky',
    #                  'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard',
    #                  'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth',
    #                  'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
    #                  'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car',
    #                  'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow',
    #                  'Car Mount', 'Ego Vehicle']
    feature_names = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane',
                     'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road',
                     'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
                     'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain', 'Sand', 'Sky',
                     'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard',
                     'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth',
                     'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
                     'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car',
                     'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow',
                     'Car Mount', 'Ego Vehicle']
    votes = pd.read_csv(votes_path, sep = '\t', header=None)
    count = 0
    length = len(votes)
    for index, row in votes.iterrows():
        target.append(choice_to_numerical(row[1]))
        left_info = get_seg(row[2])
        right_info = get_seg(row[3])
        info = (left_info - right_info).numpy()
        data.append(info)
        count += 1
        if count%5000 == 0:
            print('data loading: {}/{}'.format(count,length))
    return Bunch(data=data, target=target, feature_names=feature_names)

#def rf(att,args,dataset_train, dataset_test):
def rf(att,args,dataset):
    rfc = RandomForestClassifier(oob_score=True)  # GUO: how to choose the para? # it should be decided by the user instead of the model
    # rfc = rfc.fit(dataset_train.data, dataset_train.target)
    # acc = rfc.score(dataset_test.data, dataset_test.target)
    # return acc, oob_score
    rfc = rfc.fit(dataset.data, dataset.target)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    joblib.dump(rfc, os.path.join(args.output_dir, 'rf_{}_{}.joblib'.format(att,now)))
    oob_score = rfc.oob_score_
    return oob_score

# def test_TrainDataset(att, dataset_train):
#     rfc = joblib.load(os.path.join(args.output_dir, 'rf_{}.joblib'.format(att)))
#     acc = rfc.score(dataset_train.data, dataset_train.target)


def main(args):
    for att in args.atts:
        # dataset_train = get_dataset(os.path.join(args.dataset, '{}_train.csv'.format(att)))
        # print('training dataset of {} is loaded'.format(att))
        # dataset_test = get_dataset(os.path.join(args.dataset, '{}_test.csv'.format(att)))
        # print('test dataset of {} is loaded'.format(att))
        #acc, oob_score = rf(att, args, dataset_train, dataset_test)
        dataset = get_dataset(os.path.join(args.dataset, '{}.csv'.format(att)))
        # print('test dataset of {} is loaded'.format(att))
        oob_score = rf(att, args, dataset)
        print('The results of {}: '.format(att))
        #print('Accuracy: ', acc)
        print('Oob_score: ', oob_score)
        print('----------------------------------------------------------')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
