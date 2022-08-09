import os
import time

from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd


def read_img_as_bin(path):
    with open(path, 'rb') as f:
        img = np.asarray(f.read())
    return img


def make_debug_dataset(img_paths, output, n_images, n_dataset):
    paths = np.random.choice(img_paths, replace=False, size=n_images)
    keys = make_dataset(paths, output)
    left = np.random.choice(keys, size=n_dataset)
    right = np.random.choice(keys, size=n_dataset)
    pd.DataFrame({'left': left, 'right': right}).to_csv(output.replace('.hdf5', '.csv'))


def make_dataset(img_paths, output):
    keys = []
    with h5py.File(output, 'w') as f:
        for k, path in tqdm(enumerate(img_paths), desc='HDF5 dataset', total=len(img_paths)):
            key = os.path.basename(path)
            f.create_dataset(key, data=read_img_as_bin(path))
            keys.append(key)
    return keys


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='pp2_images_seg.hdf5')
    parser.add_argument('--img-dir', default=r'\\wsl.localhost\Ubuntu-20.04\home\xinyi\Mask2Former\output')
    parser.add_argument('--debug', action='store_true',
                        help='make a debug dataset with less images and a new votes file (see --n-img, --n-ds)')
    parser.add_argument('--n-img', type=int, default=1000, help='only used with --debug')
    parser.add_argument('--n-ds', type=int, default=12800,
                        help='only used with --debug')  # number of image pairs in the csv
    # TODO: --preprocess to normalize the image before adding to the HDF5 file
    args = parser.parse_args()

    start = time.time()
    img_paths = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir)]
    if not args.debug:
        make_dataset(img_paths, args.output)
    else:
        make_debug_dataset(img_paths, args.output, args.n_img, args.n_ds)
    print('Finished in {:.2f} s'.format(time.time() - start))