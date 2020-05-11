from skimage.filters import rank
from skimage.morphology import disk
from skimage.color import rgb2gray, rgb2hed, rgb2hsv

import numpy as np


def frac_bilateral(image, selem):
    
    pop = rank.pop((255 * image).astype(int), selem)
    pop_bilateral = rank.pop_bilateral((255 * image).astype(int), selem, s0=10, s1=10)
    
    return pop_bilateral / pop

    
ranking_fct = {"entropy": rank.entropy,
               "bottomhat": rank.bottomhat,
               "tophat" : rank.tophat,
               "mean": rank.mean,
               "bilateral":frac_bilateral}

def get_image_gray(image, image_gray):
    
    if image_gray is None:
        image_gray = rgb2gray(image)
    
    return image_gray

def get_image_hsv(image, image_hsv):
    
    if image_hsv is None:
        image_hsv = rgb2hsv(image)
    
    return image_hsv


def get_feature(image, image_gray, image_hsv, name):
    
    c = name.split('_')[0]
    
    img_c = None
    if c == 'r':
        img_c = image[:, :, 0]
    if c == 'g':
        img_c = image[:, :, 1]
    if c == 'b':
        img_c = image[:, :, 2]
    if c == 'h':
        image_hsv = get_image_hsv(image, image_hsv)
        img_c = image_hsv[:, :, 0]
    if c == 's':
        image_hsv = get_image_hsv(image, image_hsv)
        img_c = image_hsv[:, :, 1]
    if c == 'v':
        image_hsv = get_image_hsv(image, image_hsv)
        img_c = image_hsv[:, :, 2]
    if c == 'gray':
        image_gray = get_image_gray(image, image_gray)
        img_c = image_gray
    
    if img_c is None:
        raise ValueError('Unknown feature name : {}'.format(name))
    
    if len(name.split('_')) == 1:
        features = img_c
    elif len(name.split('_')) == 2:
        f = ranking_fct[name.split('_')[1]]
        r = max(int(min(image.shape) * 0.01), 1)
        selem = disk(r)
        features = f(img_c, selem)
    elif len(name.split('_')) == 3:
        f = ranking_fct[name.split('_')[1]]
        selem_frac = float(name.split('_')[2]) / 100
        r = max(int(min(image.shape) * selem_frac), 1)
        selem = disk(r)
        features = f(img_c, selem)
    else:
        raise ValueError('Unknown feature name : {}'.format(name))
    
    return features.ravel(), image_gray, image_hsv

def pixel_features(image, features_list = ['r', 'g', 'b', 'gray', 'h', 's', 'v']):
    

    all_features = []
    image_gray = None
    image_hsv = None
    
    for f in features_list:
        features, image_gray, image_hsv = get_feature(image, image_gray, image_hsv, f)
        all_features.append(features)
    
    return np.vstack(all_features).transpose()
    



def image_features(image):
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
