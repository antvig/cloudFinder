import numpy as np
from skimage.color import rgb2gray, rgb2hed, rgb2hsv
from skimage.transform import resize
import pandas as pd
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_image(image_path, from_web=False):
    
    if from_web:
        image_color = io.imread(image_path)
    else:
        try:
            image_color = io.imread(image_path + '.png')
        except FileNotFoundError:
            image_color = io.imread(image_path + '.jpg')

    return image_color/255


def resize_image(image, size):
    img_h = image.shape[0]
    img_l = image.shape[1]

    if img_h == img_l:
        image_resized = resize(image, (size, size))
    elif img_h < img_l:
        image_resized = resize(image, (size, round(size * img_l / img_h)))
    else:
        image_resized = resize(image, (round(size * img_h / img_l), size))

    return image_resized


def split_image_by_size(image, sqr_img_size=10):
    img_h = image.shape[0]
    img_l = image.shape[1]

    nbr_h_split = int(img_h / sqr_img_size)
    nbr_l_split = int(img_l / sqr_img_size)

    img_pieces_array = np.zeros((nbr_h_split * nbr_l_split, sqr_img_size, sqr_img_size, image.shape[2]))

    for i in range(nbr_h_split):
        for j in range(nbr_l_split):
            img_pieces_array[i * nbr_l_split + j] = image[i * sqr_img_size:i * sqr_img_size + sqr_img_size,
                                                    j * sqr_img_size:j * sqr_img_size + sqr_img_size]

    return img_pieces_array


def split_image_by_nbr_split(image, nbr_h_split, nbr_l_split):
    img_h = image.shape[0]
    img_l = image.shape[1]

    img_piece_h_size = int(img_h / nbr_h_split)
    img_piece_l_size = int(img_l / nbr_l_split)

    img_pieces_array = np.zeros((nbr_h_split * nbr_l_split, img_piece_h_size, img_piece_l_size, image.shape[2]))
    img_pieces_bounds = np.zeros((nbr_h_split * nbr_l_split, 4))

    for i in range(nbr_h_split):
        for j in range(nbr_l_split):
            l_l = (i * img_piece_h_size, j * img_piece_l_size)
            u_r = (i * img_piece_h_size + img_piece_h_size, j * img_piece_l_size + img_piece_l_size)
            img_pieces_array[i * nbr_l_split + j] = image[l_l[0]:u_r[0], l_l[1]:u_r[1]]
            img_pieces_bounds[i * nbr_l_split + j, :2] = l_l
            img_pieces_bounds[i * nbr_l_split + j, 2:] = u_r

    return img_pieces_array, img_pieces_bounds


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
    plt.imshow(image)
    
def get_piece(image, lb_h, lb_l, ub_h, ub_l):
    return image[lb_h:ub_h, lb_l:ub_l]
    
def plot_piece(image, lb_h, lb_l, ub_h, ub_l):

    image_piece = get_piece(image, lb_h, lb_l, ub_h, ub_l)
    plt.imshow(image_piece)
    
    
    