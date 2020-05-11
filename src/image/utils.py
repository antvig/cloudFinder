import numpy as np
from skimage.color import rgb2gray, rgb2hed, rgb2hsv, rgba2rgb

import pandas as pd
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(image_path, image_name, from_web=False):
    
    if from_web:
        image_color = io.imread(image_path)
    else:
        if image_name[-4:] in ['.png', '.jpg']:
            image_color = io.imread(image_path + image_name)
        else:
            try:
                image_color = io.imread(image_path + image_name + '.png')
            except FileNotFoundError:
                image_color = io.imread(image_path + image_name + '.jpg')
    
    if image_color.shape[-1] == 4:
        image_color = rgba2rgb(image_color)

    return (image_color/255).astype(np.float32)





def extract_multiple_image_stats(images_array):
    n_image = images_array.shape[0]

    i_mean = images_array.mean(axis=(1, 2)).reshape(n_image, 1)
    i_std = images_array.std(axis=(1, 2)).reshape(n_image, 1)
    i_max = images_array.max(axis=(1, 2)).reshape(n_image, 1)
    i_min = images_array.min(axis=(1, 2)).reshape(n_image, 1)

    image_gray = rgb2gray(images_array)

    mean_gray = image_gray.mean(axis=(1, 2)).reshape(n_image, 1)
    std_gray = image_gray.std(axis=(1, 2)).reshape(n_image, 1)
    max_gray = image_gray.max(axis=(1, 2)).reshape(n_image, 1)
    min_gray = image_gray.min(axis=(1, 2)).reshape(n_image, 1)

    features = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "r_min",
                "g_min", "b_min", "r_max", "g_max", "b_max", "gray_mean", "gray_std", "gray_min", "gray_max"]

    return np.hstack((i_mean, i_std, i_max, i_min, mean_gray, std_gray, max_gray, min_gray)), features



def extract_image_stats(image):
    r_mean = image[:, :, 0].mean()
    g_mean = image[:, :, 1].mean()
    b_mean = image[:, :, 2].mean()
    r_std = image[:, :, 0].std()
    g_std = image[:, :, 1].std()
    b_std = image[:, :, 2].std()
    r_min = image[:, :, 0].min()
    g_min = image[:, :, 1].min()
    b_min = image[:, :, 2].min()
    r_max = image[:, :, 0].max()
    g_max = image[:, :, 1].max()
    b_max = image[:, :, 2].max()

    image_gray = rgb2gray(image)

    gray_mean = image_gray.mean()
    gray_std = image_gray.std()
    gray_min = image_gray.min()
    gray_max = image_gray.max()
    
    image_hsv = rgb2hsv(image)
    h_mean = image_hsv[:, :, 0].mean()    
    s_mean = image_hsv[:, :, 1].mean()
    v_mean = image_hsv[:, :, 2].mean()
    h_std = image_hsv[:, :, 0].std()
    s_std = image_hsv[:, :, 1].std()
    v_std = image_hsv[:, :, 2].std()
    h_min = image_hsv[:, :, 0].min()
    s_min = image_hsv[:, :, 1].min()
    v_min = image_hsv[:, :, 2].min()
    h_max = image_hsv[:, :, 0].max()
    s_max = image_hsv[:, :, 1].max()
    v_max = image_hsv[:, :, 2].max()

    # data = np.array([[r_mean, g_mean, b_mean, r_std, g_std, b_std, r_min, g_min,
    #                  b_min, r_max, g_max, b_max, gray_mean, gray_std, gray_min, gray_max]])
    data = [r_mean, g_mean, b_mean, r_std, g_std, b_std, r_min, g_min, b_min, r_max, g_max, b_max, 
            gray_mean, gray_std, gray_min, gray_max,
           h_mean, s_mean, v_mean,h_std, s_std, v_std,h_min, s_min, v_min,h_max, s_max, v_max]
    features = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "r_min", "g_min", "b_min", "r_max", "g_max", "b_max", 
                "gray_mean", "gray_std", "gray_min", "gray_max",
               "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std", "h_min", "s_min", "v_min","h_max", "s_max", "v_max"]

    return data, features  # pd.DataFrame(data, columns=features)


def plot(image):
    plt.imshow(image.astype(np.float32))
    
def get_piece(image, ub_h, ub_l, lb_h, lb_l):
    return image[ub_h:lb_h, ub_l:lb_l]
    
def plot_piece(image, lb_h, lb_l, ub_h, ub_l):

    image_piece = get_piece(image, lb_h, lb_l, ub_h, ub_l)
    plt.imshow(image_piece)
    
    
    