# -*- coding: utf-8 -*-

##
## @file        convert_cifar10_to_bmp.py
## @brief       Script to Convert CIFAR-10 Dataset from Pickle Object to Bitmap
## @author      Keitetsu
## @date        2019/02/21
## @copyright   Copyright (c) 2019 Keitetsu
## @par         License
##              This software is released under the MIT License.
##

import argparse
import os
import pickle
import numpy as np
import cv2


def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'latin-1')

    return dict


def get_cifar10_dict(path_prefix, filename):
    filepath = os.path.join(path_prefix, filename)
    batch_dict = unpickle(filepath)
    print(batch_dict.keys())

    return batch_dict


def convert_cifar10_to_bmp(path_prefix, path_output, filenames):

    for filename in filenames:
        batch_dict = {}
        batch_dict = get_cifar10_dict(path_prefix, filename)
        n_images = len(batch_dict['data'])
        print("[INFO] #images@%s: %d" % (filename, n_images))

        for i in range(n_images):
            data = batch_dict['data'][i]
            filename = batch_dict['filenames'][i]

            filepath = os.path.join(path_output,  filename)

            image = data.reshape(3, 32, 32).transpose(1, 2, 0)
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, image_cv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "script to convert from CIFAR-10 dataset to bitmap"
    )
    parser.add_argument(
        '--input',
        '-i',
        type = str,
        default = "/data/rech/huiyuche/bgpt_2/cifar/train/cifar-10-batches-py",
        help = "CIFAR-10 dataset directory"
    )
    args = parser.parse_args()

    filenames_train = (
        'data_batch_1',
        # 'data_batch_2',
        # 'data_batch_3',
        # 'data_batch_4',
        # 'data_batch_5',
    )

    filenames_test = (
        'test_batch',
    )

    input_dirpath = args.input
    convert_cifar10_to_bmp(input_dirpath, '/data/rech/huiyuche/bgpt_2/cifar/train', filenames_train)
    #convert_cifar10_to_bmp(input_dirpath, 'test', filenames_test)

    # python cifar10_to_bmp.py --input cifar-10-batches-py