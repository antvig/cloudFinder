import pandas as pd
import numpy as np

from src.distribute import distribute_groupby_computation

import gc

def compute_feature_importance(traindataset, features, target, model_class):
    m = model_class()
    m.fit(traindataset[features], traindataset[target])
    features_importance = pd.DataFrame([m.feature_importances_], columns = features)
    
    return features_importance


def compute_bootstrap_feature_importance(
    bootstrap_traindataset, target, features, model_class, bootstrap_label
):
    bootstrap_features_importance = distribute_groupby_computation(
        compute_feature_importance, bootstrap_traindataset, gp_by=bootstrap_label, features=features, target=target, model_class=model_class
    )

    bootstrap_features_importance = bootstrap_features_importance.groupby('fold').mean().transpose()
    
    bootstrap_features_importance["features"] = bootstrap_features_importance.index
    bootstrap_features_importance.reset_index(inplace=True, drop=True)

    return bootstrap_features_importance

# def create_feature_df(df, nbr_split=5):
#     """

#     :param df: must contain 'id' of image
#     :return:
#     """
#     all_img_features = []
#     all_img_bounds = []
#     for idx, img in tqdm(df.iterrows()):
#         img_id = img['id']
#         image = load_image(IMAGE_PATH + img['id'])
        
#         if nbr_split > 1:
#             img_pieces_list, img_pieces_bounds = split_image_by_nbr_split(image, nbr_h_split=nbr_split, nbr_l_split=nbr_split)
#         else:
#             img_pieces_list = [image]
#             img_pieces_bounds = [image.shape[0], 0, image.shape[1], 0]
        
#         for img_piece in img_pieces_list:
#             img_piece_features, features_name = extract_image_stats(img_piece)
#             img_piece_features.append(img_id)
#             all_img_features.append(img_piece_features)
        
#         all_img_bounds.append(img_pieces_bounds)
        
#     all_img_bounds = np.vstack(all_img_bounds)

#     all_img_features = pd.DataFrame.from_records(data=all_img_features, columns=features_name + ['id'])
#     all_img_bounds = pd.DataFrame.from_records(data=all_img_bounds, columns=['lb_h', 'lb_l', 'ub_h', 'ub_l'])
    
    

#     return pd.merge(all_img_bounds, all_img_features, left_index =True, right_index=True) 
