import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import math
import requests
import zipfile
import datetime

from pathlib import Path
from random import randint, choices
from os.path import join, isdir, isfile
from distutils import dir_util
from shutil import copy
from sklearn.metrics import confusion_matrix


def create_dir(path, verbose=0):
    res = dir_util.mkpath(str(path))
    if verbose:
        return res # return all created dirs

def copytree(src, dst, create_dst=True, verbose=0):
    if create_dst:
        create_dir(str(dst))
    res = dir_util.copy_tree(str(src), str(dst))
    if verbose:
        return res # return all copied files

def get_dirs(path):
    return [name for name in os.listdir(path) if isdir(join(path, name))]

def get_files(path):
    return [name for name in os.listdir(path) if isfile(join(path, name))]

def empty(n):
    n = np.array(n)
    return (n == np.array(None)).all() or n.size == 0

def mount_gdrive(path='/content/drive', force_remount=False):
    '''
    Mount google drive and return the drive path
    '''
    from google.colab import drive
    drive.mount(path, force_remount=force_remount)
    return path

def random_imgs(data_dir, rand_imgs=5, equal_img_per_class=None, rand_classes=None, label_mode='class', label_class_names=None):
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

def random_img(data_dir, rand_classes=None, label_mode='class', label_class_names=None):
    res = random_imgs(data_dir, rand_imgs=1, rand_classes=rand_classes, label_mode=label_mode, label_class_names=label_class_names)
    return res[0][0], res[1][0] if label_mode else res[0]

def print_class_files(data_dir, print_full=False):
    data_dir = str(data_dir)
    print(f'Directory : {data_dir}')
    for i in os.listdir(data_dir):
        class_dir = join(data_dir, i)
        if isdir(class_dir):
            c = len([fl for fl in os.listdir(class_dir) if isfile(join(class_dir, fl))])
            print(f'  - Found {c} {class_dir if print_full else i}')

def load_image(file_path, image_shape=None):
    file_path = str(file_path)
    if not isfile(file_path):
        raise ValueError(f'File not found : "{file_path}"')
    img =  cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, image_shape) if image_shape else img

def load_images(file_paths, image_shape=None):
    res = []
    for file_path in file_paths:
        res.append(load_image(file_path, image_shape))
    return np.array(res)

def load_images_genarator(file_paths, image_shape=None):
    for file_path in file_paths:
        yield load_image(file_path, image_shape)

def download_image(url, download_path='', return_mode='img', image_shape=None):
    ''' return_mode : ['img', 'path', 'name', None] '''
    if return_mode not in ['img', 'path', 'name', None]:
        raise ValueError("return_mode should be one of ['img', 'path', 'name', None]")

    url = str(url)
    fl_name = url.split('/')[-1].split('?')[0]
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f'"{url}" -> returned status_code : {res.status_code}')
    if res.headers['Content-Type'] == 'text/html':
        raise Exception(f'"{url}" -> returned text/html; not an image')

    path = join(download_path, fl_name)
    with open(path, 'wb') as fl:
        fl.write(res.content)

    if return_mode == 'img':
        return load_image(path, image_shape)
    elif return_mode == 'name':
        return fl_name
    elif return_mode == 'path':
        return path
    else:
        return None

def download_images(urls, download_path='', return_mode='img', image_shape=None):
    ''' return_mode : ['img', 'path', 'name', None] '''
    r = []
    for url in urls:
        res = download_image(url, download_path, return_mode, image_shape)
        if return_mode:
            r.append(res)
    if return_mode:
        return np.array(r) if return_mode == 'img' else r

def download_images_genarator(urls, download_path='', return_mode='img', image_shape=None):
    ''' return_mode : ['img', 'path', 'name', None] '''
    for url in urls:
        yield download_image(url, download_path, return_mode, image_shape)

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

def labels_fixer(labels, var_name='labels'):
    ndim = labels.ndim
    if ndim == 2: # categorical
        return np.argmax(labels, axis=1)
    elif ndim == 1: # int
        return labels
    raise ValueError(f"'{var_name}' should be 1D (int mode) or 2D (categorical mode). Found {ndim}D")

def pred_fixer(pred, pred_mode, percent_decimal=2):
    if pred_mode == 'softmax':
        return get_pred_percent(pred, percent_decimal)
    elif pred_mode == 'sigmoid':
        return get_pred_percent_sigmoid(pred, percent_decimal)
    return pred, None

def __plot_an_image(img, title=None, rescale=None, image_shape=None, boundary=False, title_dict=dict(), plt_dict=dict()):
    if rescale:
        img = img.astype(np.float32) * rescale
    if image_shape:
        img = cv2.resize(img, image_shape)
    plt.imshow(img, **plt_dict)
    if boundary:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.axis(False)
    if title:
        plt.title(title, **title_dict)

def plot_images(imgs, labels=None, as_tuple=False, class_names=None, col=5, single_figsize=(4, 4), show_shape=False, from_link=False, from_dir=False, rescale=None, IMAGE_SHAPE=None, show_boundary=False, tight=False, save=None, title_dict=dict(), plt_dict=dict()):
    '''
    Plotting images using matplolib

    Parameters:
        imgs : array of images
        labels : labels for the images (Optional)
        as_tuple : if 'imgs' are tuples. e.g. [(img, label), (img, label), .....]
        class_names : All class_names for the images (default : None)
        col  : column number (default : 5)
        single_figsize : plot size for each img
        show_shape : define if the shape will be shown in title (default : False)
        from_link : if the imgs are links of images
        from_dir : if the imgs are paths of images
        rescale : rescale images (e.g. 1/255)
        IMAGE_SHAPE : reshapimg images
        show_boundary : show axis without ticks
        title_dict : keyword aurguments goes to plt.title()
        tight : tight layout -> plt.tight_layout()
        save : saving figure -> plt.savefig(save)
        plt_dict : keyword aurguments goes to plt.imshow()
    '''
    has_label, has_class_names = not empty(labels), not empty(class_names)

    imgs = np.array(list(imgs))
    if (from_dir or from_link) and imgs.ndim != 1:
        raise ValueError(f"'{'from_dir' if from_dir else 'from_link'}' needs 'imgs' in 1D. Found {imgs.ndim}")
    if has_label:
        labels = labels_fixer(np.array(list(labels)))

    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)

    for c, img in enumerate(zip(imgs, labels) if not as_tuple and has_label else imgs):
        title = ''
        if as_tuple or has_label:
            img, label = img
            title = f'{class_names[label]}' if has_class_names else f'{label}'
        
        if from_dir:
            img = load_image(img)
        elif from_link:
            img = download_image(img)
        if show_shape:
            title += f' {img.shape}'
        
        plt.subplot(row, col, c+1)
        __plot_an_image(img, title, rescale, IMAGE_SHAPE, show_boundary, title_dict, plt_dict)

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plot_image(img, label=None, class_names=None, figsize=(6, 6), show_shape=False, from_link=False, from_dir=False, rescale=None, IMAGE_SHAPE=None, show_boundary=False, tight=False, save=None, title_dict=dict(), plt_dict=dict()):
    plot_images(np.expand_dims(img, 0), labels=[label] if label else None, class_names=class_names, col=1, single_figsize=figsize, show_shape=show_shape, from_link=from_link, from_dir=from_dir, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, show_boundary=show_boundary, tight=tight, save=save, title_dict=title_dict, plt_dict=plt_dict)

def plot_pred_images(imgs, y_pred, y_true=None, y_pred_mode='softmax', class_names=None, col=5, single_figsize=(4, 4), show_percent=True, percent_decimal=2, rescale=None, IMAGE_SHAPE=None, show_boundary=False, title_color=('green', 'red'), tight=False, save=None, title_dict=dict(), plt_dict=dict()):
    '''
    y_pred_mode : ['softmax', 'sigmoid', 'int']
    '''
    if y_pred_mode not in ['softmax', 'sigmoid', 'int']:
        raise ValueError("y_pred_mode should be in ['softmax', 'sigmoid', 'int']")

    has_label, has_class_names = not empty(y_true), not empty(class_names)

    imgs = np.array(list(imgs))
    y_pred, percents = pred_fixer(np.array(list(y_pred)), pred_mode=y_pred_mode, percent_decimal=percent_decimal)
    if has_label:
        y_true = labels_fixer(np.array(list(y_true)), var_name='y_true')
    
    row, col, figsize = get_row_col_figsize(len(imgs), col, single_figsize)
    plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        title = ''
        if y_pred_mode != 'int' and show_percent: # we have percents
            title += f"{percents[i]}% "
        title += f"{class_names[y_pred[i]]}" if has_class_names else f"{y_pred[i]}"
        color = title_dict.get('color', 'black')
        if has_label:
            title += f" ({class_names[y_true[i]]})" if has_class_names else f" ({y_true[i]})"
            color = (title_color[0] if y_true[i] == y_pred[i] else title_color[1]) if title_color else color
        new_title_dict = title_dict.copy()
        new_title_dict['color'] = color
        
        plt.subplot(row, col, i+1)
        __plot_an_image(img, title, rescale, IMAGE_SHAPE, show_boundary, title_dict, plt_dict)

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plot_pred_image(img, y_pred, y_true=None, y_pred_mode='softmax', class_names=None, figsize=(4, 4), show_percent=True, percent_decimal=2, rescale=None, IMAGE_SHAPE=None, show_boundary=False, title_color=('green', 'red'), tight=False, save=None, title_dict=dict(), plt_dict=dict()):
    plot_pred_images(np.expand_dims(img, 0), y_pred=np.expand_dims(y_pred, 0), y_true=None if empty(y_true) else np.expand_dims(y_true, 0), y_pred_mode=y_pred_mode, class_names=class_names, col=1, single_figsize=figsize, show_percent=show_percent, percent_decimal=percent_decimal, rescale=rescale, IMAGE_SHAPE=IMAGE_SHAPE, show_boundary=show_boundary, title_color=title_color, tight=tight, save=save, title_dict=title_dict, plt_dict=plt_dict)

def plot_history(history, col=3, single_figsize=(6, 4), keys=None, fixed_xlim=False):
    epochs = history.epoch
    history = history.history
    all_keys = history.keys()

    # find errors
    if len(epochs) == 0:
        raise ValueError("The history object is empty.")
    elif len(all_keys) == 0:
        raise ValueError("The history object has no key.")

    if not empty(keys):
        for key in keys:
            if key not in all_keys:
                raise ValueError(f'"{key}" not found in the history keys')
        all_keys = keys
    
    true_keys = [i for i in all_keys if not i.startswith('val_')]
    true_plus_val_keys = true_keys + ['val_' + i for i in true_keys if 'val_' + i in all_keys]
    plot_key = true_keys + list(set(all_keys) - set(true_plus_val_keys))

    start_epoch = epochs[0] + 1
    epochs = range(start_epoch, len(epochs) + start_epoch)

    row, col, figsize = get_row_col_figsize(len(plot_key), col=col, single_figsize=single_figsize)
    plt.figure(figsize=figsize)

    for c, key in enumerate(plot_key):
        plt.subplot(row, col, c+1)
        plt.plot(epochs, history[key], label=key)
        if 'val_' + key in all_keys:
            plt.plot(epochs, history['val_' + key], label='val_' + key)
        plt.xlabel('epoch')
        if fixed_xlim:
            plt.xlim([epochs[0], epochs[-1]])
        plt.legend()
    plt.show()

def compare_histories(old, new, single_figsize=(8, 4), keys=None, fixed_xlim=False):
    """
    Compares two model history objects.
    """
    old, new, old_epochs, new_epochs = old.history, new.history, old.epoch, new.epoch
    all_keys = list(set(old.keys()) & set(new.keys())) # get all common keys

    # find errors
    if len(old_epochs) == 0:
        raise ValueError("The first history object is empty.")
    elif len(new_epochs) == 0:
        raise ValueError("The second history object is empty.")
    elif len(all_keys) == 0:
        raise ValueError("No common keys found in these two history object.")

    if not empty(keys):
        for key in keys:
            if key not in all_keys:
                raise ValueError(f'"{key}" not found in the histories common keys')
        all_keys = keys
    
    true_keys = [i for i in all_keys if not i.startswith('val_')]
    true_plus_val_keys = true_keys + ['val_' + i for i in true_keys if 'val_' + i in all_keys]
    plot_key = true_keys + list(set(all_keys) - set(true_plus_val_keys))

    start_epoch = old_epochs[0] + 1
    mid_epoch = len(old_epochs) + start_epoch - 1
    total_epochs = range(start_epoch, mid_epoch + len(new_epochs) + 1)

    row, col, figsize = get_row_col_figsize(len(plot_key), col=1, single_figsize=single_figsize)
    plt.figure(figsize=figsize)

    for c, key in enumerate(plot_key):
        plt.subplot(row, col, c+1)
        plt.plot(total_epochs, old[key] + new[key], label=key)
        if 'val_' + key in all_keys:
            plt.plot(total_epochs, old['val_' + key] + new['val_' + key], label='val_' + key)
        plt.plot([mid_epoch, mid_epoch], plt.ylim())
        plt.xlabel('epoch')
        if fixed_xlim:
            plt.xlim([total_epochs[0], total_epochs[-1]])
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
            copy(join(src, name), join(output_dir, 'train', class_name, name))
        for name in val_FileNames:
            copy(join(src, name), join(output_dir, 'val', class_name, name))
        for name in test_FileNames:
            copy(join(src, name), join(output_dir, 'test', class_name, name))

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
    from tensorflow.keras.callbacks import TensorBoard

    log_dir = join(dir_name, experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Saving TensorBoard log files to: {log_dir}")
    return TensorBoard(log_dir=log_dir)
