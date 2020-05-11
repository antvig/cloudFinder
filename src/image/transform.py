from skimage.transform import resize
import numpy as np


def get_piece(image, ub, lb):
    return image[ub[0]:lb[0], ub[1]:lb[1]]

def make_square(image, deform=False):
    
    img_h = image.shape[0]
    img_l = image.shape[1]
    
    min_size = min(img_h, img_l)
    
    if deform: 
        image_square = resize(image, (min_size, min_size))
        ub = (0, 0)
        lb = (img_h, img_l)
    else: 
        image_square = image[:min_size, :min_size]
        ub = (0, 0)
        lb = (min_size, min_size)
    
    return image_square.astype(np.float32), ub, lb
    
    
    
def resize_image(image, size):
    img_h = image.shape[0]
    img_l = image.shape[1]

    if img_h == img_l:
        image_resized = resize(image, (size, size), anti_aliasing=True)
    elif img_h < img_l:
        image_resized = resize(image, (size, round(size * img_l / img_h)), anti_aliasing=True)
    else:
        image_resized = resize(image, (round(size * img_h / img_l), size), anti_aliasing=True)

    return image_resized.astype(np.float32)


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

    return img_pieces_array.astype(np.float32)


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

    return img_pieces_array.astype(np.float16), img_pieces_bounds
