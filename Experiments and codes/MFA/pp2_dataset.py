import io
import os

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from turbojpeg import TurboJPEG, TJPF_RGB
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageFolder:
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_names = self._load_image_names(img_folder)
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        fname = self.img_names[idx]
        img = self.get_image(fname)
        return img

    def get_image(self, filename):
        img = self.load_image(filename)
        img = self.transform(img)
        return img

    def load_image(self, filename):
        return Image.open(os.path.join(self.img_folder, filename))

    def _load_image_names(self, folder):
        valid_formats = {'png', 'jpg', 'jpeg'}
        names = [fname for fname in os.listdir(folder) if self._get_image_format(fname).lower() in valid_formats]
        return names

    def _get_image_format(self, filename):
        return filename.split('.')[-1]


class PP2Dataset(torch.utils.data.Dataset):
    def __init__(self, votes_path, img_folder):
        self.votes = pd.read_csv(votes_path, header=None)
        self.img_folder = img_folder
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        vote = self.votes.iloc[index]
        left = self.get_image(vote[2])#left_filename
        right = self.get_image(vote[3])#right_filename
        target = choice_to_numerical(vote[1])#choice
        return left, right, target

    def __len__(self):
        return len(self.votes)

    def get_image(self, filename):
        img = self.load_image(filename)
        img = self.transform(img)
        return img

    def load_image(self, filename):
        return Image.open(os.path.join(self.img_folder, filename))


class PP2HDF5Dataset(torch.utils.data.Dataset):

    _LOADERS = ('pil', 'turbojpeg')

    def __init__(self, votes_path, hdf5_path, seg_raw_path, seg_sem_path, loader='pil'):
        self.votes = pd.read_csv(votes_path, sep = '\t', header=None)
        self.dataset_path = hdf5_path
        self.seg_raw_path = seg_raw_path
        self.seg_sem_path = seg_sem_path
        if loader.lower() not in self._LOADERS:
            raise ValueError("Loader must be one of {} (received {})".format(self._LOADERS, loader))
        if loader == 'tubojpeg':
            self.turbojpeg_decoder = TurboJPEG()
        self.loader = loader.lower()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.dataset_path, 'r')

    def close_hdf5(self):
        self.img_hdf5.close()

    def __del__(self):
        if hasattr(self, 'img_hdf5'):
            self.close_hdf5()

    def __getitem__(self, index: int):
        vote = self.votes.iloc[index]
        left = self.get_image(vote[2])
        right = self.get_image(vote[3])
        target = choice_to_numerical(vote[1])
        left_seg, left_info, left_area = self.get_seg(vote[2])
        right_seg, right_info, right_area = self.get_seg(vote[3])
        return left, right, left_seg, right_seg, \
               left_info.unsqueeze(0).double(), right_info.unsqueeze(0).double(), \
               left_area.unsqueeze(0).double(), right_area.unsqueeze(0).double(), \
               target

    def get_image(self, filename):
        img = self.load_image(filename)
        img = self.transform(img)
        return img

    def get_seg(self, filename):
        cp_mask = torch.load(os.path.join(self.seg_raw_path,filename+'.pt'),map_location=torch.device('cpu'))
        cp_sem = torch.load(os.path.join(self.seg_sem_path,filename+'.pt'),map_location=torch.device('cpu'))
        mask = cp_mask['mask']
        info = cp_sem['info']
        area = cp_sem['area']
        return mask, info, area

    def load_image(self, filename):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        if self.loader == 'pil':
            return Image.open(io.BytesIO(np.asarray(self.img_hdf5[filename+'.JPG'])))
        elif self.loader == 'turbojpeg':
            return self.turbojpeg_decoder.decode(self.img_hdf5[filename][()], pixel_format=TJPF_RGB)
        raise ValueError("Loader must be one of {} (received {})".format(self._LOADERS, self.loader))

    def __len__(self):
        return len(self.votes)


def choice_to_numerical(choice):
    if choice == 'left':
        return 0.
    elif choice == 'right':
        return 1.
    raise ValueError('choice must be either left or right, no other value is supported')
