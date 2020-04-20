from src.image.utils import load_image, split_image_by_nbr_split, extract_image_stats, extract_multiple_image_stats, resize_image
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

IMAGE_PATH = 'data/img/'


# def create_sample_df(df):
    
#     all_img_sample = []
#     all_img_label = []

#     for idx, img in tqdm(df.iterrows()):
#         img_id = img['id']
#         image = load_image(IMAGE_PATH + img['id'])
#         img_resized = resize_image(image, size=100)
#         img_pieces_list = split_image(img_resized, sqr_img_size=10)
        
#         for img_piece in img_pieces_list:
#             all_img_sample.append(img_piece.ravel())
#             all_img_label.append(img_id)
            
#     #all_img_sample = pd.DataFrame.from_records(data=all_img_sample)
    
#     return all_img_sample, all_img_label
        
        

    
def create_feature_df(df, nbr_split=5):
    """

    :param df: must contain 'id' of image
    :return:
    """
    all_img_features = []
    all_img_bounds = []
    for idx, img in tqdm(df.iterrows()):
        img_id = img['id']
        image = load_image(IMAGE_PATH + img['id'])
        
        if nbr_split > 1:
            img_pieces_list, img_pieces_bounds = split_image_by_nbr_split(image, nbr_h_split=nbr_split, nbr_l_split=nbr_split)
        else:
            img_pieces_list = [image]
            img_pieces_bounds = [image.shape[0], 0, image.shape[1], 0]
        
        for img_piece in img_pieces_list:
            img_piece_features, features_name = extract_image_stats(img_piece)
            img_piece_features.append(img_id)
            all_img_features.append(img_piece_features)
        
        all_img_bounds.append(img_pieces_bounds)
        
    all_img_bounds = np.vstack(all_img_bounds)

    all_img_features = pd.DataFrame.from_records(data=all_img_features, columns=features_name + ['id'])
    all_img_bounds = pd.DataFrame.from_records(data=all_img_bounds, columns=['lb_h', 'lb_l', 'ub_h', 'ub_l'])
    
    

    return pd.merge(all_img_bounds, all_img_features, left_index =True, right_index=True) 
