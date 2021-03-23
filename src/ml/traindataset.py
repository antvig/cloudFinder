import numpy as np
import pandas as pd

import tqdm

from src.distribute import distribute_groupby_computation

import gc


def create_traindataset_from_meta(df_meta, on, meta_to_X_y_fct, **kwargs):
    X = []
    y = []
    meta = []

    idx = 0
    for img, df_img in tqdm.tqdm(df_meta.groupby(on)):
        tmp_X, tmp_y, tmp_meta = meta_to_X_y_fct(df_img, **kwargs)
        if tmp_X is not None:
            X.append(tmp_X)
            y.append(tmp_y)
            tmp_meta.extend([idx])
            idx += 1
        else:
            tmp_meta.extend([-1])
        meta.append(tmp_meta)

    return (
            np.stack(X),
            np.stack(y),
            pd.DataFrame(
                    meta, columns=["img_name", "img_class", "width", "height", "sky_coverage", "is_used", "idx"]
                    )
            )


def create_traindataset_from_meta_old(
        df_meta, on, meta_to_traindataset_fct, features, distribute, **kwargs
        ):
    if distribute:
        # TODO make this work with numpy array
        traindataset, traindataset_meta = distribute_groupby_computation(
                meta_to_traindataset_fct,
                df_meta,
                gp_by=on,
                features_list=features,
                **kwargs,
                )
    else:
        traindataset = []
        traindataset_meta = []
        idx = 0
        for img, df_img in tqdm.tqdm(df_meta.groupby(on)):
            tmp, tmp_meta = meta_to_traindataset_fct(df_img)
            if tmp is None:
                tmp_meta["idx"] = -1
                traindataset_meta.append(tmp_meta)
            else:
                tmp_meta["idx"] = idx
                idx += 1
                traindataset.append(tmp)
        print("o")

    return traindataset, traindataset_meta


def create_bootstrap_traindataset_from_meta(
        df_meta,
        on,
        meta_to_traindataset_fct,
        bootsrap_nbr,
        subsample_size,
        features,
        **kwargs,
        ):
    """
    Create train dataset on subsampled data

    :param df_meta:
    :param on:
    """
    label = df_meta[on].unique()

    bootstrap_traindataset = []

    for i in range(bootsrap_nbr):
        idx_to_evaluate = np.random.randint(len(label), size=subsample_size)
        label_to_evaluate = label[idx_to_evaluate]
        df_meta_to_evaluate = df_meta[df_meta[on].isin(label_to_evaluate)]

        traindataset, traindataset_meta = distribute_groupby_computation(
                meta_to_traindataset_fct,
                df_meta_to_evaluate,
                gp_by=on,
                features_list=features,
                **kwargs,
                )
        traindataset["fold"] = i
        bootstrap_traindataset.append(traindataset.copy())

        gc.collect()

    gc.collect()

    return pd.concat(bootstrap_traindataset)
