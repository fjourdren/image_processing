# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random

# data root path
dir_dataset = 'database'

# set split ratios
validation_ratio = 0.2
test_ratio = 0.2


# get classes list
classes_dir = []
for cls in os.listdir(dir_dataset):
    classes_dir.append(cls)


# delete old train, validation & test if it exists
dir_train = os.path.join(dir_dataset, 'train')
dir_validation = os.path.join(dir_dataset, 'validation')
dir_test = os.path.join(dir_dataset, 'test')


# if dataset has already been split, we stop the process
if os.path.exists(dir_train) or os.path.exists(dir_validation) or os.path.exists(dir_test):
    print("[Error] {}, {} and {} need to be deleted before running dataset splitting process.".format(dir_train, dir_validation, dir_test))
    exit()


print("-" * 24)


# init stats
nbClass = 0
nbImages = 0
train_nbImages = 0
validation_nbImages = 0
test_nbImages = 0


# create train, validation & test directories
for cls in classes_dir:
    os.makedirs(os.path.join(dir_train, cls))
    os.makedirs(os.path.join(dir_validation, cls))
    os.makedirs(os.path.join(dir_test, cls))


    # folder to copy images from
    src = os.path.join(dir_dataset, cls)

    allfilenames = os.listdir(src) # get list of images
    np.random.shuffle(allfilenames) # randomize the array


    # split dataset with split ratios
    train_filenames, validation_filenames, test_filenames = np.split(np.array(allfilenames),
                                                            [int(len(allfilenames) * (1 - (validation_ratio + test_ratio))), 
                                                            int(len(allfilenames) * (1 - test_ratio))])


    # make source path to be copied
    train_filenames = [os.path.join(src, name) for name in train_filenames.tolist()]
    validation_filenames   = [os.path.join(src, name) for name in validation_filenames.tolist()]
    test_filenames  = [os.path.join(src, name) for name in test_filenames.tolist()]

    # copy pasting images in the good directory
    for name in train_filenames:
        shutil.copy(name, os.path.join(dir_train, cls))

    for name in validation_filenames:
        shutil.copy(name, os.path.join(dir_validation, cls))

    for name in test_filenames:
        shutil.copy(name, os.path.join(dir_test, cls))


    # display class stats to user
    print('* Class', cls)
    print('     - Images:', len(allfilenames))
    print('     - Training images:', len(train_filenames))
    print('     - Validation images:', len(validation_filenames))
    print('     - Test images:', len(test_filenames))
    print("-" * 24)

    # increase global stats
    nbClass += 1
    nbImages += len(allfilenames)
    train_nbImages += len(train_filenames)
    validation_nbImages += len(validation_filenames)
    test_nbImages += len(test_filenames)


# display final dataset stats to user
print('\n\n==== Final dataset state ====')
print('     - Classes:', nbClass)
print('     - Total images:', nbImages)
print('     - Total training images:', train_nbImages)
print('     - Total validation images:', validation_nbImages)
print('     - Total test images:', test_nbImages)
print("=" * 29)