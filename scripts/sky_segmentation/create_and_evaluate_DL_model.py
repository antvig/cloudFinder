import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
os.chdir(project_path)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.sky_segmentation.sky_segmentation import get_sun_image_X_y_DL
from src.ml.traindataset import create_traindataset_from_meta
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from numpy import save, load
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser("Create and Evaluate DL model for sky segmentation")
parser.add_argument("model", type=str, help="model name")
parser.add_argument("-r", "--recreate_dataset", help="Create dataset even if already existing", action='store_true')

Y_LABEL = "y_dl.npy"
Y_LABEL_PRED = "y_pred_dl.npy"
X_LABEL = "X_dl.npy"
DATASET_META_LABEL = "dataset_meta.csv"
DATA_PATH = 'data/process/sky_segmentation'
MODEL_PATH = "models/sky_segmentation"

BACKBONE = 'efficientnetb0'

if __name__ == "__main__":

    args = parser.parse_args()
    model_name = args.model
    recreate_dataset = args.recreate_dataset

    # INIT
    model_path = os.path.join(MODEL_PATH, model_name)
    if os.path.isdir(model_path) and not model_name == "tmp":
        raise PermissionError('{} already exist ! '.format(model_path))
    else:
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    # I - Load metadata
    print('-- load metadata')
    df_sun_meta = pd.read_csv("data/img_metadata/sun.csv")

    # II - Create X y
    if (not recreate_dataset) & os.path.isfile(os.path.join(DATA_PATH, X_LABEL)) & os.path.isfile(
            os.path.join(DATA_PATH, Y_LABEL)):
        print('-- load dataset')
        X = load(os.path.join(DATA_PATH, X_LABEL))
        y = load(os.path.join(DATA_PATH, Y_LABEL))
        dataset_meta = pd.read_csv(os.path.join(DATA_PATH, DATASET_META_LABEL))

    else:
        print('-- create dataset')
        X, y, dataset_meta = create_traindataset_from_meta(
                df_sun_meta, on="img_name", size=64, meta_to_X_y_fct=get_sun_image_X_y_DL
                )
        save(os.path.join(DATA_PATH, X_LABEL), X)
        save(os.path.join(DATA_PATH, Y_LABEL), y)
        dataset_meta.to_csv(os.path.join(DATA_PATH, DATASET_META_LABEL), index=False)

    # # III - Evaluate and learn model
    print('-- Evaluate and learn model')
    cv = KFold(5, shuffle=True)
    fold = 0
    fold_pred = []
    fold_meta = []
    for (idx_train, idx_test) in cv.split(X, y):
        print('---- fold {}'.format(fold))
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        dataset_meta_train = dataset_meta[dataset_meta.idx.isin(idx_train)].copy()
        dataset_meta_test = dataset_meta[dataset_meta.idx.isin(idx_test)].copy()

        model = Unet(BACKBONE, classes=1, activation="sigmoid", encoder_weights='imagenet')
        model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

        history = model.fit(
                x=X_train,
                y=y_train.astype(float),
                batch_size=8,
                epochs=10,
                validation_data=(X_test, y_test.astype(float))
                )

        fold_path = os.path.join(model_path, "model_fold_{}".format(fold))
        model.save_weights(fold_path)

        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(model_path,"fold_{}.png".format(fold)))
        plt.close()

        y_predicted = model.predict(X_test)
        fold_pred.append(y_predicted)
        fold_meta.append(dataset_meta_test)
        fold += 1

    import numpy as np

    y_predicted = np.concatenate(fold_pred)
    unorted_meta = pd.concat(fold_meta)
    sorted_index = np.argsort(unorted_meta.idx)
    y_predicted = y_predicted[sorted_index]

    save(os.path.join(DATA_PATH, Y_LABEL_PRED), y_predicted)