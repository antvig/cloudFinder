import pandas as pd
from nltk.tokenize import word_tokenize

PATH = 'data/'


def ca_main_clouds(df_clouds_name):
    """
    Find the main clouds in the images from the title.
    """

    df_ca = pd.read_csv(PATH + 'img_metadata/cloud_atlas.csv')
    clouds_name = df_clouds_name['cloud_name'].to_list()
    for cloud in clouds_name:
        df_ca[cloud.lower()] = df_ca.apply(lambda x: cloud.lower() in word_tokenize(str(x['title']).lower()), axis=1)
    df_ca['source'] = 'ca'
    return df_ca[['id', 'source'] + clouds_name]


def co_main_clouds(df_clouds_name):
    """
    Find the main clouds in the images from the path.
    """

    df_co = pd.read_csv(PATH + 'img_metadata/cloud_online.csv')

    clouds_name = df_clouds_name['cloud_name'].to_list()
    for cloud in clouds_name:
        df_co[cloud.lower()] = df_co.apply(lambda x: str(x['cloud_type']).lower() == cloud.lower(), axis=1)
    df_co['source'] = 'co'

    return df_co[['id', 'source'] + clouds_name]


def cc_main_clouds(df_clouds_name):
    """
    Find the main clouds in the images from the path.
    """

    df_cc = pd.read_csv(PATH + 'img_metadata/ccsn.csv')

    for i, r in df_clouds_name.iterrows():
        cloud_name = r['cloud_name']
        cloud_abreviation = r['cloud_abreviation']
        df_cc[cloud_name.lower()] = df_cc.apply(
            lambda x: str(x['path'].split('/')[1]).lower().find(cloud_abreviation.lower()) != -1,
            axis=1)
    df_cc['source'] = 'cc'

    return df_cc[['id', 'source'] + df_clouds_name['cloud_name'].to_list()]


def create_main_clouds_label_df():
    """
    Create the label dataframe.
    :return:
    """
    df_clouds_name = pd.read_csv(PATH + "coud_labels.csv")

    ca_label = ca_main_clouds(df_clouds_name)
    co_label = co_main_clouds(df_clouds_name)
    cc_label = cc_main_clouds(df_clouds_name)

    df = pd.concat([ca_label, co_label, cc_label], axis=0, ignore_index=False).dropna(axis=1).reset_index(drop=True)

    return df
