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

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def get_dirs(path):
    return [name for name in os.listdir(path) if isdir(join(path, name))]

def get_files(path):
    return [name for name in os.listdir(path) if isfile(join(path, name))]

def empty(n):
    n = np.array(n)
    return (n == np.array(None)).all() or n.size == 0

def mount_gdrive(path='/content/drive'):
    '''
    Mount google drive and return the drive path
    '''
    from google.colab import drive
    drive.mount(path)
    return join(path, 'My drive')

def get_random_imgs(data_dir, rand_imgs=5, equal_img_per_class=None, rand_classes=None, label_mode='class', label_class_names=None):
    '''
    label_mode : 'class', 'int', None
    label_class_names : used for label_mode : 'int' (To get the index)
    '''
    data_dir = str(data_dir)
    class_names = get_dirs(data_dir)

    if label_mode not in ['class', 'int', None]:
        raise ValueError(f'label_mode : "{label_mode}" not found in ["class", "int", None]')
    if label_mode == 'int' and not label_class_names:
        raise ValueError(f"""'label_class_names' needed for 'label_mode' : "int" """)

    if rand_classes:
        for class_name in rand_classes:
            if class_name not in class_names:
                raise ValueError(f'"{class_name}" not found in "{data_dir}""')
    else:
        rand_classes = class_names

    if equal_img_per_class:
        rand_list = {class_name : equal_img_per_class for class_name in rand_classes}
    else:
        rand_list = {class_name : 0 for class_name in rand_classes}
        for class_name in choices(rand_classes, k=rand_imgs):
            rand_list[class_name] += 1

    rand = []
    labels = []
    for class_name, rand_img_num in rand_list.items():
        if rand_img_num:
            class_dir = join(data_dir, class_name)
            rand_images = choices([fl for fl in os.listdir(class_dir) if isfile(join(class_dir, fl))],
                                  k=rand_img_num)
            for i in rand_images:
                rand.append(join(class_dir, i))
                if label_mode == 'class':
                    labels.append(class_name)
                elif label_mode == 'int':
                    labels.append(label_class_names.index(class_name))

    return [rand, labels] if label_mode else rand

def print_class_files(data_dir, print_full=False):
    data_dir = str(data_dir)
    print(f'Directory : {data_dir}')
    for i in os.listdir(data_dir):
        class_dir = join(data_dir, i)
        if isdir(class_dir):
            c = len([fl for fl in os.listdir(class_dir) if isfile(join(class_dir, fl))])
            print(f'  - Found {c} {class_dir if print_full else i}')

def __image_to_numpy(file_dir, image_size):
    img =  cv2.cvtColor(cv2.imread(file_dir), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, image_size) if image_size else img

def load_images(file_dir, image_size=None):
    single = type(file_dir) == str
    if single:
        file_dir = [file_dir]

    r = []
    for fl in file_dir:
        r.append(__image_to_numpy(str(fl), image_size))
    return r[0] if single else np.array(r)

def load_images_genarator(file_dir, image_size=None):
    if type(file_dir) == str:
        file_dir = [file_dir]
    
    for fl in file_dir:
        yield __image_to_numpy(str(fl), image_size)

def __download_images(url, download_path, return_mode, image_shape):
    fl_name = url.split('/')[-1].split('?')[0]
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f'"{url}" returned status_code : {res.status_code}')
    if res.headers['Content-Type'] == 'text/html':
        raise Exception(f'"{url}" returned text/html; not an image')

    path = join(download_path, fl_name) if download_path else fl_name
    with open(path, 'wb') as fl:
        fl.write(res.content)

    if return_mode == 'img':
        return load_images(path, image_shape)
    elif return_mode == 'name':
        return fl_name
    elif return_mode == 'path':
        return path
    else:
        return None

def download_images(url, download_path=None, return_mode='img', image_shape=None):
    '''
    Parameters:
        url : just a url or list of urls
        download_path : path for downloaded img
        return_mode : ['img', 'path', 'name', None]
    '''

    if return_mode not in ['img', 'path', 'name', None]:
        raise ValueError("return_mode should be one of ['img', 'path', 'name', None]")

    single = type(url) == str
    urls = [url] if single else url

    r = []
    for url in urls:
        res = __download_images(str(url), download_path, return_mode, image_shape)
        if return_mode:
            r.append(res)

    if return_mode:
        return r[0] if single else (np.array(r) if return_mode == 'img' else r)

def download_images_genarator(url, download_path=None, return_mode='img', image_shape=None):
    '''
    Parameters:
        url : just a url or list of urls
        download_path : path for downloaded img
        return_mode : ['img', 'path', 'name', None]
    '''

    if return_mode not in ['img', 'path', 'name', None]:
        raise ValueError("return_mode should be one of ['img', 'path', 'name', None]")

    single = type(url) == str
    urls = [url] if single else url

    for url in urls:
        yield __download_images(str(url), download_path, return_mode, image_shape)

def get_pred_percent(preds, percent_round=2):
    percents = []
    ints = []
    for x in preds:
        pos = np.argmax(x)
        ints.append(pos)
        percents.append(x[pos] * 100)
    return np.array(ints), np.round(percents, percent_round)

def get_pred_percent_sigmoid(preds, percent_round=2):
    preds = np.array(preds)
    preds = preds.reshape(preds.shape[0]) # make it 1D
    percents = []
    ints = []
    for x in preds:
        pos = round(x)
        ints.append(pos)
        if pos == 0:
            x = (1 - x)
        percents.append(x * 100)
    return np.array(ints), np.round(percents, percent_round)

def get_row_col_figsize(total_item, col, single_figsize):
    col = col if total_item >= col else total_item
    row = math.ceil(total_item / col)
    figsize = (single_figsize[0]*col, single_figsize[1]*row)
    return row, col, figsize

def plot_images(imgs, labels=None, class_names=None, col=5, label_mode='int', single_figsize=(4, 4), show_shape=False, from_link=False, from_dir=False, rescale=None, IMAGE_SHAPE=None, show_boundary=False, **keyargs):
    '''
    Plotting images using matplolib

    Parameters:
        imgs : array of images
        labels : labels for the images (Optional)
        class_names : All class_names for the images (default : None)
        col  : column number (default : 5)
        label_mode : 'int' or 'categorical'
        single_figsize : plot size for each img
        show_shape : define if the shape will be shown in title (default : False)
        from_link : if the imgs are links of images
        from_dir : if the imgs are paths of images
        rescale : rescale images (e.g. 1/255)
        IMAGE_SHAPE : reshapimg images
        show_boundary : show axis without ticks
        **keyargs : extra keyword aurguments goes to pl.imshow()
    '''
    if not empty(labels):
        if label_mode == 'categorical':
            labels = np.argmax(labels, axis=1)
        elif label_mode != 'int':
            raise ValueError('label_mode shoud be "int" or "categorical"')
    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)

    if from_dir:
        imgs = load_images_genarator(imgs)
    elif from_link:
        imgs = download_images_genarator(imgs)
    else:
        imgs = np.array(imgs)

    for c, img in enumerate(imgs):
        img_shape = img.shape
        if rescale:
            img = img * rescale
        if IMAGE_SHAPE:
            img = cv2.resize(img, IMAGE_SHAPE)
        
        plt.subplot(row, col, c+1)
        plt.imshow(img, **keyargs)
        if show_boundary:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis(False)
        title = ''
        if not empty(labels): 
            if not empty(class_names):
                title = f'{class_names[labels[c]]}'
            else:
                title = f'{labels[c]}'
        if show_shape:
            title += f' {img_shape}'
        if title:
            plt.title(title)
    plt.show()

def plot_image(img, label=None, class_names=None, label_mode='int', figsize=(6, 6), show_shape=False, from_link=False, from_dir=False, rescale=None, IMAGE_SHAPE=None, show_boundary=False, **keyargs):
    '''
    Plotting an image using matplolib

    Parameters:
    ----------
        img : the image in numbers
        label : label for the image (Optional)
        class_names : All class_names for the images (default : None)
        label_mode : 'int' or 'categorical'
        figsize : Figure size for the image (default : (6, 6))
        show_shape : define if the shape will be shown in title (default : False)
        from_link : if the img is a link of image
        from_dir : if the img is a path of image
        rescale : rescale image (e.g. 1/255)
        IMAGE_SHAPE : reshapimg image
        show_boundary : show axis without ticks
        label_mode : 'int' or 'categorical'
        **keyargs : extra keyword aurguments goes to pl.imshow()
    '''
    plot_images(np.expand_dims(img, 0), labels=[label] if label else None, class_names=class_names, col=1, label_mode=label_mode, single_figsize=figsize, show_shape=show_shape, from_link=from_link, from_dir=from_dir, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, show_boundary=show_boundary, **keyargs)

def plot_pred_images(imgs, y_pred, y_true=None, y_pred_mode='softmax', y_true_mode='int', class_names=None, col=5, single_figsize=(4, 4), show_percent=True, percent_decimal=2, rescale=None, IMAGE_SHAPE=None, show_boundary=False, title_color=('green', 'red'), **keyargs):
    '''
    y_pred_mode : ['softmax', 'sigmoid', 'int']
    y_true_mode : ['categorical', 'int']
    '''

    y_pred = np.array(y_pred)
    if y_pred_mode == 'softmax':
        y_pred, percents = get_pred_percent(y_pred, percent_decimal)
    elif y_pred_mode == 'sigmoid':
        y_pred, percents = get_pred_percent_sigmoid(y_pred, percent_decimal)
    elif y_pred_mode != 'int':
        raise ValueError("y_pred_mode should be in ['softmax', 'sigmoid', 'int']")

    if not empty(y_true):
        if y_true_mode == 'categorical':
            y_true = np.argmax(y_true, axis=1)
        elif y_true_mode != 'int':
            raise ValueError('y_true_mode shoud be "int" or "categorical"')
    
    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        if rescale:
            img = img * rescale
        if IMAGE_SHAPE:
            img = cv2.resize(img, IMAGE_SHAPE)
        
        plt.subplot(row, col, i+1)
        plt.imshow(img, **keyargs)
        if show_boundary:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis(False)
        
        title = ''
        if y_pred_mode != 'int' and show_percent: # we have percents
            title += f"{percents[i]}% "

        if class_names:
            title += f"{class_names[y_pred[i]]}"
        else:
            title += f"{y_pred[i]}"

        color = 'black'
        if not empty(y_true):
            if class_names:
                title += f" ({class_names[y_true[i]]})"
            else:
                title += f" ({y_true[i]})"
            correct_pred = y_true[i] == y_pred[i]
            color = (title_color[0] if correct_pred else title_color[1]) if title_color else None
        
        plt.title(title, color=color)
    plt.show()

def plot_pred_image(img, y_pred, y_true=None, y_pred_mode='softmax', y_true_mode='int', class_names=None, figsize=(4, 4), show_percent=True, percent_decimal=2, rescale=None, IMAGE_SHAPE=None, show_boundary=False, title_color=('green', 'red'), **keyargs):
    plot_pred_images(np.expand_dims(img, 0), y_pred=np.expand_dims(y_pred, 0), y_true=None if empty(y_true) else np.expand_dims(y_true, 0), y_pred_mode=y_pred_mode, y_true_mode=y_true_mode, class_names=class_names, col=1, single_figsize=figsize, show_percent=show_percent, percent_decimal=percent_decimal, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, show_boundary=show_boundary)

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

def plot_confusion_matrix(y_test, y_pred, class_names, figsize=(10, 7), fontsize=14, xticks_rotation=45, title=None):
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
    print(f"Saving TensorBoard log files to: {log_dir}")
    return TensorBoard(log_dir=log_dir)

def create_modelcheckpoint_callback(checkpoint_path, **args):
    return ModelCheckpoint(filepath=checkpoint_path, **args)
