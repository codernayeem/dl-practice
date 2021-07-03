import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import shutil
import math
import requests

from pathlib import Path
from random import randint, choice
from os.path import join
from sklearn.metrics import confusion_matrix


def download_a_image(url):
    fl_name = url.split('/')[-1]
    res = requests.get(url)
    with open(fl_name, 'wb') as fl:
        fl.write(res.content)
    return image_to_numpy(fl_name)

def image_to_numpy(file_dir, image_size=None):
    img =  cv2.cvtColor(cv2.imread(file_dir), cv2.COLOR_BGR2RGB)
    return img if not image_size else cv2.resize(img, image_size)

def get_class_percent(pred, classes):
    pred = pred.squeeze()
    class_int = np.argmax(pred)
    return classes[class_int], pred[class_int] * 100

def plot_images(imgs, labels=None, col=5, classes=None, figsize=(10, 6), show_shape=False, gray=False, single_plot_size=None):
    '''
    Plotting images using matplolib

    Parameters:
        imgs : array of images
        labels : labels for the images (Optional)
        col  : column number (default : 5)
        classes : All classes for the images (default : None)
        figsize : Figure size (default : (10, 6))
        show_shape : define if the shape will be shown in title (default : False)
        gray : set True to set gray colormap
        single_plot_size : if given, figsize will be calculated using this
    '''
    if gray:
        plt.gray()
    row = math.ceil(len(imgs) / col)
    if single_plot_size:
        figsize = (single_plot_size[0]*col, single_plot_size[1]*row)
    plt.figure(figsize=figsize)
    for c, img in enumerate(imgs):
        plt.subplot(row, col, c+1)
        plt.imshow(img)
        plt.axis(False)
        title = ''
        if labels is not None and classes is not None:
            title = f'{classes[labels[c]]}'
        elif labels is not None:
            title = f'{labels[c]}'
        if show_shape:
            title += f' ({img.shape})'
        if title:
            plt.title(title)
    plt.show()

def plot_image(img, label=None, classes=None, figsize=(6, 6), show_shape=False, gray=False):
    '''
    Plotting an image using matplolib

    Parameters:
    ----------
        img : the image in numbers
        label : label for the image (Optional)
        classes : All classes for the images (default : None)
        figsize : Figure size for the image (default : (6, 6))
        show_shape : define if the shape will be shown in title (default : False)
        gray : set True to set gray colormap
    '''
    plot_images(np.expand_dims(img, 0), labels=None if label == None else [label], col=1, classes=classes, figsize=figsize, show_shape=show_shape, gray=gray)

def plot_pred_images(model, imgs, labels=None, col=5, figsize=(10, 6), classes=None, rescale=None, IMAGE_SHAPE=None, from_links=False, from_dirs=False, gray=False, single_plot_size=None):
    if gray:
        plt.gray()
    row = math.ceil(len(imgs) / col)
    if single_plot_size:
        figsize = (single_plot_size[0]*col, single_plot_size[1]*row)
    plt.figure(figsize=figsize)
    for c, img in enumerate(imgs):
        if from_dirs:
            img = image_to_numpy(img)
        elif from_links:
            img = download_a_image(img)

        plt.subplot(row, col, c+1)
        plt.imshow(img)
        plt.axis(False)
        if IMAGE_SHAPE:
            img = cv2.resize(img, IMAGE_SHAPE)
        if rescale:
            img *= rescale
        class_name, percent = get_class_percent(model.predict(np.expand_dims(img, axis=0)), classes)
        title = f'{round(percent, 2)}% {class_name}'
        if labels is not None:
            title += f' ({classes[labels[c]]})'
        plt.title(title)
    plt.show()

def plot_pred_image(model, img, label=None, figsize=(6, 6), classes=None, rescale=None, IMAGE_SHAPE=None, from_link=False, from_dir=False, gray=False):
    plot_pred_images(model, np.expand_dims(img, 0), labels=None if label == None else [label], col=1, figsize=figsize, classes=classes, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, from_links=from_link, from_dirs=from_dir, gray=gray, single_plot_size=None)

def plot_history(history, val=True, togather_all=False, figsize=(10, 6), accuracy='accuracy'):
    history = pd.DataFrame(history.history)
    if togather_all:
        history.plot(xlabel='Epochs')
    else:
        loss = ['loss']
        acc = [accuracy]
        if val:
            loss.append('val_loss')
            acc.append(f'val_{accuracy}')
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        history[loss].plot(xlabel='Epochs', y_label='Loss')
        plt.subplot(1, 2, 2)
        history[acc].plot(xlabel='Epochs', y_label=accuracy.title())

def create_train_val_test(root_data_dir, val_ratio=0.1, test_ratio=0, output_dir=None, class_labels=None):
    # class labels
    if not class_labels:
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

