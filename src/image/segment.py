from skimage.draw import polygon2mask
import numpy as np

def get_mask(image, polygones):
    
    mask = np.zeros((image.shape[0], image.shape[1])).astype(bool)
    
    for p in polygones:    
        if len(p) == 0:
            continue
        tmp = extract_segment(image, p)
        mask = mask | tmp
    
    return mask

def get_border(image):
    
    return np.concatenate((image[:,0][::-1], image[0, 1:], image[1:,-1], image[-1,1:-1][::-1]))

def extract_segment(image, polygone_coordinate):
    
    return polygon2mask((image.shape[0], image.shape[1]), polygone_coordinate)