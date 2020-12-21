from src.image.features import pixel_features
from src.image.utils import load_image
from src.image.transform import resize_image
from src.image.segment import get_mask, get_border
from skimage.segmentation import clear_border

from skimage.morphology import opening, disk, closing

from src.data_source.sun import parse_xml_polygone

import numpy as np
import pandas as pd

from tqdm import tqdm

IMAGE_PATH = "data/img/sky_segmentation/"

# def img_correct_prediction_v2(img_prediction, features_meta):
#     """
#     Get the 
#     """
#     img_prediction_with_correction = img_prediction.copy()
    
#     size_h = features_meta[features_meta.img_name == img_name]["size_h"].iloc[0]
#     size_l = features_meta[features_meta.img_name == img_name]["size_l"].iloc[0]
    
#     sky_mask_predicted = img_prediction["is_sky_PREDICTED"].values.reshape(
#         (size_h, size_l)
#     )
    
#     upp
    
#     img_border = get_border(img_prediction["is_sky_PROBA"].values)
    
#     contour = array_img[0, :] +  array_img[0, :]
    
    
def img_correct_prediction(img_prediction, features_meta, opening_factor=4):

    img_prediction_with_correction = img_prediction.copy()

    img_name = img_prediction.loc[img_prediction.index[0], "img_name"]
    size_h = features_meta[features_meta.img_name == img_name]["size_h"].iloc[0]
    size_l = features_meta[features_meta.img_name == img_name]["size_l"].iloc[0]

    sky_mask_predicted = img_prediction["is_sky_PREDICTED"].values.reshape(
        (size_h, size_l)
    )

    sky_mask_predicted_corrected = closing(sky_mask_predicted, disk(opening_factor))
    sky_mask_predicted_corrected = sky_mask_predicted_corrected & ~clear_border(sky_mask_predicted_corrected)

    img_prediction_with_correction[
        "is_sky_PREDICTED_COR"
    ] = sky_mask_predicted_corrected.ravel().astype(int)

    return img_prediction_with_correction


def sun_meta_to_train_dataset(
    sun_img_meta, size=100, features_list=["r"], minimum_sky_coverage=0.05, download=False
):
    
    if download:
        img_path = sun_img_meta.iloc[0].img_name_path
    else:
        img_path = IMAGE_PATH
        
    img_name = sun_img_meta.iloc[0].img_name
    polygones = [parse_xml_polygone(a) for a in sun_img_meta.polygone]

    image = load_image(img_path, img_name)

    target = get_img_target(image, polygones, size)

    sky_coverage = target.sum() / len(target)

    if sky_coverage < minimum_sky_coverage:

        meta_list = [[0, 0, sky_coverage, False]]
        dataset = pd.DataFrame(columns=features_list + ["is_sky"])

    else:

        dataset, size_h, size_l = get_img_features(image, size, features_list)

        dataset = pd.DataFrame(dataset, columns=features_list)
        dataset["is_sky"] = target.astype(int)

        meta_list = [[size_h, size_l, sky_coverage, True]]

    meta = pd.DataFrame(
        meta_list, columns=["size_h", "size_l", "sky_coverage", "is_used"]
    )

    return dataset, meta


def get_img_target(img, polygones, size, ravel):

    sky_mask = get_mask(img, polygones)
    sky_mask_resized = resize_image(sky_mask.astype(np.float), size=size) > 0.5

    if ravel:
        return sky_mask_resized.ravel()
    else:
        return sky_mask_resized


def get_img_features(img, size, features_list):

    image_resized = resize_image(img, size=size)
    img_features = pixel_features(image_resized, features_list=features_list)

    return img_features, image_resized.shape[0], image_resized.shape[1]
