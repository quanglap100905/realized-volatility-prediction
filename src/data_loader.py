import pandas as pd
from joblib import Parallel, delayed

def process_all_files(file_paths, func):
    """Executes feature extraction functions concurrently across all CPU cores"""
    df_list = Parallel(n_jobs=-1, verbose=1)(delayed(func)(path) for path in file_paths)
    return pd.concat(df_list, ignore_index=True)