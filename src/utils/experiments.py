import numpy as np


def shuffle_phases(mat):
    """mat: shape (N,T) with N = mini-batch size and T = period length"""
    pass


def conduct_recurrent_training():
    pass


def get_bh_integral(df):
    """Given the B and H curve in the pandas DataFrame df, calculate the area within the polygon"""
    b, h = (
        df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy(),
        df.loc[:, [f"H_t_{k}" for k in range(1024)]].to_numpy(),
    )
    h += 100
    return (
        df.freq
        * 0.5
        * np.abs(np.sum(b * (np.roll(h, 1, axis=1) - np.roll(h, -1, axis=1)), axis=1))
    )  # shoelace formula


def get_stratified_fold_indices(df, k):
    """Given a Pandas Dataframe df, return a Pandas Series with the kfold labels for the test set.
    The test set labels are distributed such that they are stratified along the B-field's peak-2-peak value.
    The labels are an enumeration of the test sets, e.g., for a 4-fold training each row will be labeled in [0, 3].
    
    Example:
    --------
    >>> ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    >>> kfold_lbls = get_stratified_fold_indices(ds.query("material == '3F4'"), 4)  # 4-fold
    >>> kfold_lbls
    0       0
    1       2
    2       0
    3       2
    4       2
        ..
    6558    1
    6559    0
    6560    0
    6561    1
    6562    3
    Name: kfold_tst_lbl, Length: 6563, dtype: int64
    >>> ds.loc[:, 'kfold'] = kfold_lbls
    >>> for i in range(4):
    ...     test_ds = ds.query("kfold == @i")
    ...     train_ds = ds.query("kfold != @i")
            # train on this fold
    """
    full_b = df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()

    df = (
        df.assign(b_peak2peak=full_b.max(axis=1) - full_b.min(axis=1))
        .reset_index(drop=False, names="orig_idx")
        .sort_values("b_peak2peak", ascending=True)
        .assign(kfold_tst_lbl=np.tile(np.arange(k), np.ceil(len(df) / k).astype(int))[: len(df)])
        .sort_values('orig_idx', ascending=True)
    )
    return df.kfold_tst_lbl

def form_factor(b): 
    """
    definition:      kf = rms / mean(abs)
    for ideal sine:  np.pi/(2*np.sqrt(2))
    """
    return np.sqrt(np.mean(b**2, axis=1))  / np.mean(np.abs(b), axis=1)

def crest_factor(b): 
    """
    definition:      ks = rms / max()
    for ideal sine:  np.sqrt(2) 
    """
    return np.max(np.abs(b), axis=1)  / np.sqrt(np.mean(b**2, axis=1))
