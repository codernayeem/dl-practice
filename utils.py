import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import shutil
import math
import requests
import zipfile
import datetime

from pathlib import Path
from random import randint, choices
from os.path import join, isdir, isfile
from sklearn.metrics import confusion_matrix

import tensorflow as tf


def empty(n):
    n = np.array(n)
    return (n == np.array(None)).all() or n.size == 0

def mount_gdrive():
    '''
    Mount google drive and return drive path '/content/drive/My Drive'
    '''
    from google.colab import drive
    drive.mount('/content/drive')
    return '/content/drive/My Drive'

def categorical_to_int(labels, axis=1):
    '''
    return labels in int mode from categorical mode

    Parameters:
    -----------
        labels : labels in categorical mode
        axis : if labels are in 2D set axis=1 or if labels are in 1D set axis=0
    '''
    return np.argmax(labels, axis=axis)

def get_random_imgs(data_dir, rand_imgs=5, equal_img_per_class=None, classes=None, return_labels=True, label_classes=None, label_mode='class'):
    # label_mode : 'class', 'int'
    # label_classes : used for label_mode : 'int' (To get the index)
    data_dir = str(data_dir)
    class_names = [class_name for class_name in os.listdir(data_dir) if isdir(join(data_dir, class_name))]

    if return_labels:
        if label_mode not in ['class', 'int']:
            raise ValueError(f'label_mode : "{label_mode}" not found in ["class", "int"]')
        if label_mode == 'int' and not label_classes:
            raise ValueError(f"""'label_classes' needed for 'label_mode' : "int" """)

    if classes:
        for class_name in classes:
            if class_name not in class_names:
                raise ValueError(f'"{class_name}" not found in "{data_dir}""')
    else:
        classes = class_names

    if equal_img_per_class:
        rand_list = {class_name : equal_img_per_class for class_name in classes}
    else:
        rand_list = {class_name : 0 for class_name in classes}
        for class_name in choices(classes, k=rand_imgs):
            rand_list[class_name] += 1

    rand = []
    labels = []
    for class_name, rand_img_num in rand_list.items():
        if rand_img_num:
            class_dir = join(data_dir, class_name)
            rand_images = choices([fl for fl in os.listdir(class_dir) if isfile(join(class_dir, fl))],
                                  k=rand_img_num)
            for i in rand_images:
                rand.append(join(data_dir, class_name, i))
                if return_labels:
                    if label_mode == 'class':
                        labels.append(class_name)
                    else:
                        labels.append(label_classes.index(class_name))

    return (rand, labels) if return_labels else rand

def print_class_files(data_dir, print_full=False):
    data_dir = str(data_dir)
    print(f'Directory : {data_dir}')
    for i in os.listdir(data_dir):
        class_dir = join(data_dir, i)
        if isdir(class_dir):
            c = len([fl for fl in os.listdir(class_dir) if isfile(join(class_dir, fl))])
            print(f'  - Found {c} {class_dir if print_full else i}')

def download_a_image(url):
    fl_name = url.split('/')[-1]

    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f'"{url}" returned status_code : {res.status_code}')
    if res.headers['Content-Type'] == 'text/html':
        raise Exception(f'"{url}" returned text/html; not an image')
    
    with open(fl_name, 'wb') as fl:
        fl.write(res.content)
    return image_to_numpy(fl_name)

def image_to_numpy(file_dir, image_size=None):
    img =  cv2.cvtColor(cv2.imread(file_dir), cv2.COLOR_BGR2RGB)
    return img if not image_size else cv2.resize(img, image_size)

def get_class_percent(pred, classes=None):
    pred = pred.squeeze()
    class_int = np.argmax(pred)
    return (class_int if empty(classes) else classes[class_int]), pred[class_int] * 100

def get_row_col_figsize(total_item, col, single_figsize):
    col = col if total_item >= col else total_item
    row = math.ceil(total_item / col)
    figsize = (single_figsize[0]*col, single_figsize[1]*row)
    return row, col, figsize

def plot_images(imgs, labels=None, col=5, classes=None, label_mode='int', single_figsize=(5, 5), show_shape=False, from_link=False, from_dir=False, cmap=None, show_boundary=False):
    '''
    Plotting images using matplolib

    Parameters:
        imgs : array of images
        labels : labels for the images (Optional)
        col  : column number (default : 5)
        classes : All classes for the images (default : None)
        single_figsize : plot size for each img
        show_shape : define if the shape will be shown in title (default : False)
        cmap : color map for imgs (e.g. cmap=plt.cm.gray)
        show_boundary : show axis without ticks
        label_mode : 'int' or 'categorical'
    '''
    if not empty(labels):
        if label_mode == 'categorical':
            labels = categorical_to_int(labels)
        elif label_mode != 'int':
            raise ValueError('label_mode shoud be "int" or "categorical"')
    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)
    for c, img in enumerate(imgs):
        if from_dir:
            img = image_to_numpy(img)
        elif from_link:
            img = download_a_image(img)
        plt.subplot(row, col, c+1)
        plt.imshow(img, cmap=cmap)
        if show_boundary:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis(False)
        title = ''
        if not empty(labels): 
            if not empty(classes):
                title = f'{classes[labels[c]]}'
            else:
                title = f'{labels[c]}'
        if show_shape:
            title += f' ({img.shape})'
        if title:
            plt.title(title)
    plt.show()

def plot_image(img, label=None, classes=None, label_mode='int', figsize=(6, 6), show_shape=False, from_link=False, from_dir=False, cmap=None, show_boundary=False):
    '''
    Plotting an image using matplolib

    Parameters:
    ----------
        img : the image in numbers
        label : label for the image (Optional)
        classes : All classes for the images (default : None)
        figsize : Figure size for the image (default : (6, 6))
        show_shape : define if the shape will be shown in title (default : False)
        cmap : color map for img (e.g. cmap=plt.cm.gray)
        show_boundary : show axis without ticks
        label_mode : 'int' or 'categorical'
    '''
    plot_images(np.expand_dims(img, 0), labels=None if label == None else [label], col=1, classes=classes, label_mode=label_mode, single_figsize=figsize, show_shape=show_shape, from_link=from_link, from_dir=from_dir, cmap=cmap, show_boundary=show_boundary)

def plot_pred_images(model, imgs, labels=None, col=5, label_mode='int', classes=None, single_figsize=(5, 5), rescale=None, IMAGE_SHAPE=None, from_link=False, from_dir=False, cmap=None, show_boundary=False):
    if not empty(labels):
        if label_mode == 'categorical':
            labels = categorical_to_int(labels)
        elif label_mode != 'int':
            raise ValueError('label_mode shoud be "int" or "categorical"')
    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)
    for c, img in enumerate(imgs):
        if from_dir:
            img = image_to_numpy(img)
        elif from_link:
            img = download_a_image(img)

        plt.subplot(row, col, c+1)
        plt.imshow(img, cmap=cmap)
        if show_boundary:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis(False)
        if IMAGE_SHAPE:
            img = cv2.resize(img, IMAGE_SHAPE)
        if rescale:
            img = img * rescale
        class_name, percent = get_class_percent(model.predict(np.expand_dims(img, axis=0)), classes)
        title = f'{round(percent, 2)}% {class_name}'
        if not empty(labels):
            title += f' ({classes[labels[c]]})'
        plt.title(title)
    plt.show()

def plot_pred_image(model, img, label=None, label_mode='int', classes=None, figsize=(6, 6), rescale=None, IMAGE_SHAPE=None, from_link=False, from_dir=False, cmap=None, show_boundary=False):
    plot_pred_images(model, np.expand_dims(img, 0), labels=None if label == None else [label], col=1, label_mode=label_mode, classes=classes, single_figsize=figsize, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, from_link=from_link, from_dir=from_dir, cmap=cmap, show_boundary=show_boundary)

def plot_history(history, col=3, single_figsize=(6, 4), keys=None, start_epoch=1):
    history = history.history
    all_keys = history.keys()

    if not empty(keys):
        for key in keys:
            if key not in all_keys:
                raise ValueError(f'"{key}" not found in the history keys')
        all_keys = keys
    
    true_keys = [i for i in all_keys if not i.startswith('val_')]
    true_plus_val_keys = true_keys + ['val_' + i for i in true_keys if 'val_' + i in all_keys]
    plot_key = true_keys + list(set(all_keys) - set(true_plus_val_keys))

    total_epochs = range(start_epoch, len(history[plot_key[0]])+start_epoch)

    row, col, figsize = get_row_col_figsize(len(plot_key), col=col, single_figsize=single_figsize)
    plt.figure(figsize=figsize)

    for c, key in enumerate(plot_key):
        plt.subplot(row, col, c+1)
        plt.plot(total_epochs, history[key], label=key)
        if 'val_' + key in all_keys:
            plt.plot(total_epochs, history['val_' + key], label='val_' + key)
        plt.xlabel('epoch')
        plt.legend()
    plt.show()

def compare_histories(old, new, initial_epochs, single_figsize=(8, 4), keys=None, start_epoch=1):
    """
    Compares two model history objects.
    """
    old, new = old.history, new.history
    all_keys = old.keys()
    
    if not empty(keys):
        for key in keys:
            if key not in all_keys:
                raise ValueError(f'"{key}" not found in the history keys')
        all_keys = keys
    
    true_keys = [i for i in all_keys if not i.startswith('val_')]
    true_plus_val_keys = true_keys + ['val_' + i for i in true_keys if 'val_' + i in all_keys]
    plot_key = true_keys + list(set(all_keys) - set(true_plus_val_keys))

    total_epochs = range(start_epoch, len(old[plot_key[0]]) + len(new[plot_key[0]])+start_epoch)

    row, col, figsize = get_row_col_figsize(len(plot_key), col=1, single_figsize=single_figsize)
    plt.figure(figsize=figsize)

    for c, key in enumerate(plot_key):
        plt.subplot(row, col, c+1)
        plt.plot(total_epochs, old[key] + new[key], label=key)
        if 'val_' + key in all_keys:
            plt.plot(total_epochs, old['val_' + key] + new['val_' + key], label='val_' + key)
        plt.plot([initial_epochs, initial_epochs], plt.ylim())
        plt.xlabel('epoch')
        plt.legend()
    plt.show()

def create_train_val_test(root_data_dir, val_ratio=0.1, test_ratio=0, output_dir=None, class_labels=None):
    # class labels
    if empty(class_labels):
        class_labels = os.listdir(root_data_dir)
    
    # output_dir
    if not output_dir:
        output_dir = root_data_dir

    # creating folders
    try:
        for class_name in class_labels:
            os.makedirs(join(output_dir, 'train', class_name))
            if val_ratio:
                os.makedirs(join(output_dir, 'val', class_name))
            if test_ratio:
                os.makedirs(join(output_dir, 'test', class_name))
    except:
        pass

    for class_name in class_labels:
        src = join(root_data_dir, class_name)
        all_files = os.listdir(src)
        np.random.shuffle(all_files)

        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(all_files), [
            int(len(all_files) * (1 - (val_ratio + test_ratio))),
            int(len(all_files) * (1 - test_ratio)),
            ])
        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(join(src, name), join(output_dir, 'train', class_name, name))
        for name in val_FileNames:
            shutil.copy(join(src, name), join(output_dir, 'val', class_name, name))
        for name in test_FileNames:
            shutil.copy(join(src, name), join(output_dir, 'test', class_name, name))

def print_confusion_matrix(y_test, y_pred, class_names, figsize=(10, 7), fontsize=14, xticks_rotation=45, title=None):
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=xticks_rotation, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.title(title)
    plt.show()

def unzip_data(filename, unzip_dir=None):
  """
  Unzips filename into the current working / given directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall(unzip_dir)
  zip_ref.close()

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"

    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = join(dir_name, experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback
