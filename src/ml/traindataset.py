import numpy as np
import pandas as pd

from src.distribute import distribute_groupby_computation

import gc


def create_traindataset_from_meta(
    df_meta, on, meta_to_traindataset_fct, features, **kwargs
):
    traindataset, traindataset_meta = distribute_groupby_computation(
        meta_to_traindataset_fct,
        df_meta_to_evaluate,
        gp_by=on,
        features_list=features,
        **kwargs,
    )

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
