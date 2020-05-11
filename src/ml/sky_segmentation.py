from src.image.features import pixel_features
from src.image.utils import load_image
from src.image.transform import make_square, resize_image
from src.image.segment import extract_segment

from src.data_source.sun import find_polygone_coordinate

import numpy as np
import pandas as pd

from tqdm import tqdm

IMAGE_PATH = 'data/img/sky_segmentation/'



    


def create_feature_df(df_meta, size = 100, deform=True, features_list = ['r'], minimum_sky_coverage= 0.05, _use_progress_bar=False):
    """
    :return:
    """
    
    
    all_img_features = []
    all_img_features_meta = []
    
    for img_name in tqdm(set(df_meta.img_name), disable=not _use_progress_bar):
        
        img_meta = df_meta[df_meta.img_name == img_name].copy()
        
        img_features, img_features_meta = img_feature(img_meta, 
                                                      size=size, 
                                                      deform=deform, 
                                                      features_list=features_list,
                                                     minimum_sky_coverage=minimum_sky_coverage)
        
        all_img_features.append(img_features)
        all_img_features_meta.append(img_features_meta)

        

    all_img_features = pd.concat(all_img_features)
    all_img_features_meta = pd.concat(all_img_features_meta)
    
    return all_img_features, all_img_features_meta


def img_feature(img_meta, size=100, deform=True, features_list=['r'], minimum_sky_coverage= 0.05):
    img_path = img_meta.iloc[0].img_name_path
    img_name = img_meta.iloc[0].img_name

    image = load_image(img_path, img_name)

    sky_mask = get_sky_mask(image, img_meta)
    sky_coverage = sky_mask.sum() / sky_mask.size

    if sky_coverage < minimum_sky_coverage:
        img_features_meta = pd.DataFrame(data=[[img_name,
                                                img_path,
                                                0,
                                                0,
                                                sky_coverage,
                                                0,
                                                False]],
                                         columns=['img_name',
                                                  'img_path',
                                                  'size_h',
                                                  'size_l',
                                                  'sky_coverage',
                                                  'sky_coverage_after_resize',
                                                  'is_used'])

        img_features = None

    else:

        image_resized = resize_image(image, size=size)
        sky_mask_resized = resize_image(sky_mask.astype(np.float), size=size) > 0.5

        img_features = pixel_features(image_resized, features_list=features_list)

        img_features = pd.DataFrame(data=img_features, columns=features_list)
        img_features['img_name'] = img_name
        img_features['is_sky'] = sky_mask_resized.ravel()
        
        sky_coverage_resize = sky_mask_resized.sum() / sky_mask_resized.size

        img_features_meta = pd.DataFrame(data=[[img_name,
                                                img_path,
                                                image_resized.shape[0],
                                                image_resized.shape[1],
                                                sky_coverage,
                                                sky_coverage_resize,
                                                True]],
                                         columns=['img_name',
                                                  'img_path',
                                                  'size_h',
                                                  'size_l',
                                                  'sky_coverage',
                                                  'sky_coverage_after_resize',
                                                  'is_used'])

    return img_features, img_features_meta
    
    
    
    

def get_sky_mask(image, img_meta):
    
    sky_mask = np.zeros((image.shape[0], image.shape[1])).astype(bool)
    
    for segment_idx, segment in img_meta.iterrows():
        try:
            polygone_coordinate = find_polygone_coordinate(segment.polygone)
            tmp = extract_segment(image, polygone_coordinate)
            sky_mask = sky_mask | tmp
        except:
            continue
    
    return sky_mask
    