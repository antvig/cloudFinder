# TODO not running anymore

import os, sys
from os.path import dirname
project_path = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
os.chdir(project_path)

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.offline as po

from src.sky_segmentation.sky_segmentation import (
    sun_meta_to_train_dl_dataset,
    img_correct_prediction,
)
from src.utils import reduce_mem_usage

from src.ml.score import compute_classification_score

import feather as ft
import gc

import pickle

import matplotlib.pyplot as plt

gc.collect()

####################################
# VAR
####################################

PERFORM_FEATURES_SELECTION = False
CREATE_FEATURES = True
CREATE_MODEL = False
CORRECT_MODEL = True
COMPUTE_PERFORMANCE = True

# TO STORE
MODEL_PATH = "data/process/sky_segmentation/"

TARGET = "is_sky"
NBR_FEATURES_TO_KEEP = 20
RESIZE = 100

param = {"target": TARGET, "resize": RESIZE}

# OTHER PARAM
DOWNLOAD = False

####################################
# LOAD DATA
####################################

df_sun_meta = pd.read_csv("data/img_metadata/sun.csv")
df_sun_meta = df_sun_meta.head(100)
df_sun_meta.head()

img_name = df_sun_meta.img_name.unique()

####################################
# Create Features
####################################

features_to_keep = ["r", "g", "b"]
param["features"] = features_to_keep

if CREATE_FEATURES:
    from src.ml.traindataset import create_traindataset_from_meta

    traindataset, traindataset_meta = create_traindataset_from_meta(
            df_sun_meta,
            "img_name",
            sun_meta_to_train_dl_dataset,
            features_to_keep,
            size=RESIZE,
            download=DOWNLOAD,
            distribute=False
            )

    traindataset.is_sky = traindataset.is_sky.astype(int)  # Bug

    traindataset = reduce_mem_usage(traindataset)

    ft.write_dataframe(traindataset, MODEL_PATH + "/traindataset.ft")
    ft.write_dataframe(traindataset_meta, MODEL_PATH + "/traindataset_meta.ft")

elif CREATE_MODEL:
    traindataset = ft.read_dataframe(MODEL_PATH + "/traindataset.ft")
    traindataset_meta = ft.read_dataframe(MODEL_PATH + "/traindataset_meta.ft")
else:
    traindataset_meta = ft.read_dataframe(MODEL_PATH + "/traindataset_meta.ft")


####################################
# Create Model
####################################

if CREATE_MODEL:

    from src.ml.models import VotingClassifier
    from src.ml.model_selection import ShuffleGroupKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    cv = ShuffleGroupKFold(5)
    cv_split = cv.split(traindataset.img_name)

    model = VotingClassifier()

    X = traindataset[features_to_keep].values
    y = traindataset["is_sky"].astype('int').values

    based_estimator = DecisionTreeClassifier(max_depth=10)
    y_pred = model.fit_predict_cv(X, y, cv_split, base_estimator=based_estimator)

    model_cv = traindataset[["img_name", "is_sky"]].copy()
    model_cv[["is_sky_PREDICTED", "is_sky_PROBA", "fold"]] = y_pred
    model_cv["is_sky_PREDICTED"] = model_cv["is_sky_PREDICTED"].astype(int)
    model_cv = reduce_mem_usage(model_cv)

    ft.write_dataframe(model_cv, MODEL_PATH + "/cv.ft")
    with open(MODEL_PATH + "/model_1.pkl", "wb") as f:
        pickle.dump(model, f)

else:
    model_cv = ft.read_dataframe(MODEL_PATH + "/cv.ft")
    model_cv["is_sky_PREDICTED"] = model_cv["is_sky_PREDICTED"].astype(int)