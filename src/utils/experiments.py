import numpy as np
import copy
from pathlib import Path

DATA_SOURCE = Path.cwd().parent / "data" / "input" / "raw"
PROC_SOURCE = DATA_SOURCE.parent / "processed"

# bsat map
BSAT_MAP = {
    "3C90": 0.47,
    "3C94": 0.47,
    "3E6": 0.46,
    "3F4": 0.41,
    "77": 0.51,
    "78": 0.48,
    "N27": 0.50,
    "N30": 0.38,
    "N49": 0.49,
    "N87": 0.49,
}

def shuffle_phases(mat):
    """mat: shape (N,T) with N = mini-batch size and T = period length"""
    pass


def conduct_recurrent_training():
    pass


def get_bh_integral(df):
    """Given the B and H curve in the pandas DataFrame df, calculate the area within the polygon"""
    # offset polygon into first quadrant
    b, h = (
        df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy() + 0.5,  # T
        df.loc[:, [f"H_t_{k}" for k in range(1024)]].to_numpy() + 300,  # A/m
    )
    return (
        df.freq
        * 0.5
        * np.abs(np.sum(b * (np.roll(h, 1, axis=1) - np.roll(h, -1, axis=1)), axis=1))
    )  # shoelace formula

def get_bh_integral_from_two_mats(freq, b, h):
    """ b and h are shape (#samples, #timesteps)"""
    # offset b and h into first quadrant
    h_with_offset = h + 300 # A/m
    b_with_offset = b + 0.5 # T
    return freq.ravel() * 0.5 * np.abs(np.sum(b_with_offset * (np.roll(h_with_offset, 1, axis=1) - np.roll(h_with_offset, -1, axis=1)), axis=1))

def get_stratified_fold_indices(df, n_folds):
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
        .assign(
            kfold_tst_lbl=np.tile(np.arange(n_folds), np.ceil(len(df) / n_folds).astype(int))[: len(df)])
        .sort_values("orig_idx", ascending=True)
    )
    return df.kfold_tst_lbl


def form_factor(x):
    """
    definition:      kf = rms(x) / mean(abs(x))
    for ideal sine:  np.pi/(2*np.sqrt(2))
    """
    return np.sqrt(np.mean(x**2, axis=1)) / np.mean(np.abs(x), axis=1)


def crest_factor(x):
    """
    definition:      kc = rms(x) / max(x)
    for ideal sine:  np.sqrt(2)
    """
    return np.max(np.abs(x), axis=1) / np.sqrt(np.mean(x**2, axis=1))



def bool_filter_sine(b, rel_kf=0.01, rel_kc=0.01, rel_0_dev=0.1):
    """
    b: input flux density (nxm)-array with n m-dimensional flux density waveforms
    rel_kf: (allowed) relative deviation of the form factor for sine classification
    rel_kc: (allowed) relative deviation of the crest factor for sine classification
    rel_0_dev: (allowed) relative deviation of the first value from zero (normalized on the peak value)
    """
    kf_sine = np.pi / (2 * np.sqrt(2))
    kc_sine = np.sqrt(2)

    filter_bool = [True] * b.shape[0]

    statements = [
        list(form_factor(b) < kf_sine * (1 + rel_kf)),  # form factor based checking
        list(form_factor(b) > kf_sine * (1 - rel_kf)),  # form factor based checking
        list(crest_factor(b) < kc_sine * (1 + rel_kc)), # crest factor based checking
        list(crest_factor(b) > kc_sine * (1 - rel_kc)), # crest factor based checking
        list(b[:, 0] < np.max(b, axis=1) * rel_0_dev),  # starting value based checking
        list(b[:, 0] > -np.max(b, axis=1) * rel_0_dev), # starting value based checking
    ]

    for statement in statements:
        filter_bool = [a and zr for a, zr in zip(filter_bool, statement)]

    

    return filter_bool

def bool_filter_triangular(b, rel_kf=0.005, rel_kc=0.005):
    kf_triangular = 2/np.sqrt(3)
    kc_triangular = np.sqrt(3)

    filter_bool = [True] * b.shape[0]

    statements = [list(form_factor(b) < kf_triangular * (1 + rel_kf)),
                  list(form_factor(b) > kf_triangular * (1 - rel_kf)),
                  list(crest_factor(b) < kc_triangular * (1 + rel_kc)),
                  list(crest_factor(b) > kc_triangular * (1 - rel_kc))]

    for statement in statements:
        filter_bool = [a and zr for a, zr in zip(filter_bool, statement)]

    return filter_bool

def get_waveform_est(full_b):
    """From Till's tp-1.4.7.3.1 NB, return waveform class"""
  
    # labels init all with 'other'
    k = np.zeros(full_b.shape[0], dtype=int)
    
    # square
    k[np.all(np.abs(full_b[:, 250:500:50] - full_b[:, 200:450:50]) / np.max(np.abs(full_b), axis=1).reshape(-1, 1) < 0.05, axis=1) & np.all(full_b[:, -200:]< 0, axis=1)] = 1
    
    # triangular
    k[bool_filter_triangular(full_b, rel_kf=0.01, rel_kc=0.01)] = 2

    # sine
    k[bool_filter_sine(full_b, rel_kf=0.005, rel_kc=0.005)] = 3

    return k