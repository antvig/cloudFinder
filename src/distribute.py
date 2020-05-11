from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
import numpy as np
from tqdm import tqdm

N_CORE = 5


def ditribut_groupby_computation(fct, df, gp_by , **kwargs):
    
    df_list = [gp[1] for gp in df.groupby(gp_by)]
    
    f = lambda x: fct(x, **kwargs)
    
    with Pool(N_CORE) as p:
        res = list(tqdm(p.imap(f,df_list), total=len(df_list)))
        
              
    return aggregate_res(res)

def ditribute_split_computation(fct, df, df_split=10, **kwargs):
    
    df_list = np.array_split(df, df_split)
    
    f = lambda x: fct(x, **kwargs)
    
    with Pool(N_CORE) as p:
        res = list(tqdm(p.imap(f,df_list), total=len(df_list)))
        
              
    return aggregate_res(res)


def ditribute_CV(fct, df, cv_split, **kwargs):
    
    df_list = [(train_index, test_index)  for train_index, test_index in cv_split]
    
    f = lambda x: fct(x, **kwargs)
    
    with Pool(N_CORE) as p:
        res = list(tqdm(p.imap(f,df_list), total=len(df_list)))
        
              
    return aggregate_res(res)

def aggregate_res(res):
    
    if type(res[0]) in (list, tuple):
        nbr_out = len(res[0])
        out = []

        for i in range(nbr_out):
            to_concat = [sub_res[i] for sub_res in res]
            out.append(pd.concat(to_concat))

        return tuple(out)
    else:
        return pd.concat(res)