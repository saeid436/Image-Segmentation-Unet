# training_data_preparation:
    # resizing the training images
    # attaching mask images to create one image containing all features
    # Creating X_train and Y_train...

# test_data_preparation:
    # resizing the test images
    # Creating X_test

import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

def training_data_preparation(_imgHeight, _imgWidth, _imgChannel, _trainPath):
    train_ids = next(os.walk(_trainPath))[1]

    X_train = np.zeros((len(train_ids), _imgHeight, _imgWidth, _imgChannel), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), _imgHeight, _imgWidth, 1), dtype=np.bool)

    # Building X_train and Y_train:
    print('Resizing Training Images...')
    for n , id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = _trainPath + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:_imgChannel]
        img = resize(img, (_imgHeight, _imgWidth), mode='constant', preserve_range=True)
        X_train[n] = img # Fill X_train by resized images
        mask = np.zeros((_imgHeight, _imgWidth,1), dtype=np.bool)
        for mask_file in next(os.walk(path +'/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (_imgHeight, _imgWidth), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    print('Resizing the training images DONE!!')
    return X_train, Y_train


def test_data_preparation(_imgHeight, _imgWidth, _imgChannel, _testPath):
    test_ids = next(os.walk(_testPath))[1]
    X_test = np.zeros((len(test_ids), _imgHeight, _imgWidth, _imgChannel), dtype=np.uint8)
    sizes_test = []
    # Building X_test:
    print('Resizing the Test Images...')
    for n , id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = _testPath + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:_imgChannel]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (_imgHeight, _imgWidth), mode='constant', preserve_range=True)
        X_test[n] = img # Fill X_test by resized images
    print('DONE!!')
    return X_test

