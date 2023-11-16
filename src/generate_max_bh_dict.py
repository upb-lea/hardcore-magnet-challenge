import os.path

from utils.experiments import get_waveform_est, BSAT_MAP
from run_cnn import B_COLS, H_COLS, construct_tensor_seq2seq
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Note:
# This file generates a lookup-dict for b and h normalization to the maximum value.
# so the VAL_SOURCE = DATA_SOURCE.parent / 'validation_full_dataset'
# must be the full dataset of the hosts!

DATA_SOURCE = Path.cwd().parent / 'data' / 'input' / 'raw'
VAL_SOURCE = DATA_SOURCE.parent / 'validation_full_dataset'
PROC_SOURCE = DATA_SOURCE.parent/ "processed"
PREDS_SINK = PROC_SOURCE.parent.parent / 'output'
MODELS_SINK = PREDS_SINK.parent / 'models' / '2023-10-20_No_Limit'
JSON_OUT = Path.cwd().parent / 'data'

b_max_dict = {}
h_max_dict = {}


# Load all sample datas
print("Read original data in..")
data_d = {}
for p in VAL_SOURCE.glob("*"):
    if p.name != ".gitkeep":
        print(p.name)
        data_d[p.name] = {
            f.stem: pd.read_csv(f, index_col=None) for f in p.glob("*.csv")
        }


# store compact data set
col_translation = {
    "Volumetric_Loss": "ploss",
    "H_Waveform": "H",
    "B_waveform": "B",
    "Temperature": "temp",
    "Frequency": "freq",
}
dfs = []
print("Convert data..")
for k, v in data_d.items():
    series_l = []
    for q, arr in v.items():
        if arr.shape[-1] == 1:
            df = pd.Series(arr.to_numpy().ravel(), name=col_translation[q])
        else:
            df = pd.DataFrame(
                arr.to_numpy(),
                columns=[f"{col_translation[q]}_t_{j}" for j in range(arr.shape[1])],
            )
        series_l.append(df)
    mat_df = pd.concat(series_l, axis=1)
    dfs.append(mat_df.assign(material=k))

val_ds = pd.concat(dfs, ignore_index=True)

for count_model, material_name in enumerate(['3C90', '3C94', '3E6', '3F4', '77', '78', 'Material A', 'Material B', 'Material C', 'Material D', 'Material E', 'N27', 'N30', 'N49', 'N87']):
    ds = val_ds.query(f"material == '{material_name}'").reset_index(drop=True)
    waveforms = get_waveform_est(
        ds.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()
    )

    # construct tensors
    x_cols = [
        c
        for c in ds
        if c not in ["ploss", "kfold", "material"] and not c.startswith(("B_t_", "H_t_"))
    ]
    b_limit = np.abs(ds.loc[:, B_COLS].to_numpy()).max()  # T
    h_limit = min(
        np.abs(ds.loc[:, H_COLS].to_numpy()).max(), 150
    )  # A/m

    b_max_dict[material_name] = b_limit
    h_max_dict[material_name] = h_limit

with open(os.path.join(JSON_OUT, f"b_max_dict.json"), "w") as outfile:
    json.dump(b_max_dict, outfile, indent=2)
with open(os.path.join(JSON_OUT, f"h_max_dict.json"), "w") as outfile:
    json.dump(h_max_dict, outfile, indent=2)
