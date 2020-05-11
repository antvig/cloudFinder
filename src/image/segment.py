from skimage.draw import polygon2mask




def extract_segment(image, polygone_coordinate):
    
    return polygon2mask((image.shape[0], image.shape[1]), polygone_coordinate)