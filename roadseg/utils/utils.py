import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as album


def visualize(**images): # **images
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()): #items ?
        plt.subplot(1, n_images, idx+1)
        plt.xticks([]); plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20) # str.replace.title() ?
        plt.imshow(image) # imshow ??
    plt.show()


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format 
    by replacing each pixel value with a vector of length num_classes
    # Args
        label: the 2D arrey segmentation image label
        label_values
        
    # Rets
        A 2D array with the same width and height as the input, 
        but with a depth size of num_classes
    """
    semantic_map = []
    for clr in label_values:
        equality = np.equal(label, clr)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hor format (depth is num_classes),
    to a 2D array with only 1 channer, where each pixel value is 
    the classified class key.
    # Args
        image: the one-hot format image
    
    # Rets
        A 2D array with the same size width and height as the input,
        but with a depth size of 1, where each pixel value is the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x


def color_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, color code the segmentation results.
    # Args
        image: single channel array where each value represents the class key.
        label_values
    
    # Rets
        Color coded image for segmentation visualization
    """
    color_codes = np.array(label_values)
    x = color_codes[image.astype(int)]
    return x


def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=.5),
        album.VerticalFlip(p=.5),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32') # ???


def get_preprocessing(preprocessing_fn=None):
    """
    Construct preprocessing transform
    # Args
        preprocessing_fn (callable): data normalization function
    
    # Rets
        transform: album.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)