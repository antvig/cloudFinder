from src.image.features import pixel_features
from src.image.utils import load_image
from src.image.transform import resize_image
from src.image.segment import get_mask, get_border
from skimage.segmentation import clear_border
import warnings
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
    sky_mask_predicted_corrected = sky_mask_predicted_corrected & ~clear_border(
        sky_mask_predicted_corrected
    )

    img_prediction_with_correction[
        "is_sky_PREDICTED_COR"
    ] = sky_mask_predicted_corrected.ravel().astype(int)

    return img_prediction_with_correction


def get_sun_image_X_y_DL(
    sun_img_meta,
    size=100,
    minimum_sky_coverage=0.05,
    download=False,
):
    if download:
        img_path = sun_img_meta.iloc[0].img_name_path
    else:
        img_path = IMAGE_PATH

    img_name = sun_img_meta.iloc[0].img_name
    img_class = sun_img_meta.iloc[0].img_name_path.split("/")[-2]

    # I - Load Image
    image = load_image(img_path, img_name)

    # II - Get ground truth
    polygones = [parse_xml_polygone(a) for a in sun_img_meta.polygone]
    sky_mask = get_mask(image, polygones)
    sky_coverage = sky_mask.sum() / sky_mask.size

    # III - Check the minimum sky coverage
    if sky_coverage < minimum_sky_coverage:
        meta = [
            img_name,
            img_class,
            image.shape[0],
            image.shape[1],
            sky_coverage,
            False,
        ]
        X, y = None, None
    else:
        # IV - resize Image and sky_mask
        X = resize_image(image, size, how="square")
        y = (
            resize_image(sky_mask.astype(np.float), size=size, how="square") > 0.5
        ).astype(int)
        meta = [img_name, img_class, image.shape[0], image.shape[1], sky_coverage, True]

    return X, y, meta
