import pandas as pd

from src.ml.label import create_main_clouds_label_df
from src.ml.features import create_feature_df

df_label = create_main_clouds_label_df()
df_features = create_feature_df(df_label.iloc[:2], nbr_split=0)