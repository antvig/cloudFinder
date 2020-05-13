from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
import numpy as np
from tqdm import tqdm

N_CORE = 5

def add_column(df, label, value):
    
    if len(df) == 0:
        df_to_return = pd.DataFrame(columns=list(df.columns) + [label])
    else:
        df_to_return = df.copy()
        df_to_return[label] = value
    
    return df_to_return

def f(x, fct, on, **kwargs):
    key = x[0]
    df = x[1]
    res = fct(df, **kwargs)
    res_with_label = []
    
    if type(res) != tuple:
        res = (res,)
        
    for r in res:
        if type(r) == pd.DataFrame:
            r_with_label = add_column(r, on, key)
        elif type(r) == pd.Series:
            r_with_label = r.to_frame()
            r_with_label[on] = key
        else:
            r_with_label = {key:r}
        res_with_label.append(r_with_label)
    
    if len(res_with_label) == 1:
        return res_with_label[0]
    else:
        return tuple(res_with_label)
    
    
def distribute_groupby_computation(fct_to_compute, df, gp_by, **kwargs):
    
    df_list = [(gp[0], gp[1]) for gp in df.groupby(gp_by)]
    
    f_wrapper = lambda x: f(x, fct=fct_to_compute, on=gp_by, **kwargs)
    
    with Pool(N_CORE) as p:
        res = tuple(tqdm(p.imap(f_wrapper,df_list), total=len(df_list)))
      
    return aggregate_res(res)

def ditribute_split_computation(fct, df, df_split=10, **kwargs):
    
    df_list = np.array_split(df, df_split)
    
    f = lambda x: fct(x, **kwargs)
    
    with Pool(N_CORE) as p:
        res = list(tqdm(p.imap(f,df_list), total=len(df_list)))
        
    return aggregate_res(res)


def ditribute_cv_computation(fct, df, cv_label_col, **kwargs):
    
    cv_label = df[cv_label_col].unique()
    
    df_list = [(df[df[cv_label_col] != i].index, df[df[cv_label_col] == i].index)  for i in cv_label]
    
    f_wrapper = lambda x: fct(x[0], x[1], df, **kwargs)
    
    with Pool(N_CORE) as p:
        res = list(tqdm(p.imap(f_wrapper, df_list), total=len(df_list)))
        
    return aggregate_res(res)


def aggregate_res(res):
    if type(res[0]) == tuple:
        nbr_out = len(res[0])
        out = []
        for i in range(nbr_out):
            to_concat = [sub_res[i] for sub_res in res]
            out.append(pd.concat(to_concat))

        return tuple(out)
    else:
        return pd.concat(res)