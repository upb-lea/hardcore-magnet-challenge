"""This script can be called initially to convert the original data set submission into
a more compact format, on which modeling will build upon. """
import pandas as pd
from pathlib import Path
from utils.experiments import DATA_SOURCE, PROC_SOURCE


data_d = {}

print("Read original data in..")
for p in DATA_SOURCE.glob("*"):
    if p.name != ".gitkeep":
        print(p.name)
        data_d[p.name] = {
            f.stem: pd.read_csv(f, index_col=None) for f in p.glob("*.csv")
        }


# store compact data set
col_translation = {
    "Volumetric_losses[Wm-3]": "ploss",
    "H_waveform[Am-1]": "H",
    "B_waveform[T]": "B",
    "Temperature[C]": "temp",
    "Frequency[Hz]": "freq",
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
    mat_df.to_pickle(PROC_SOURCE / f"{k}.pkl.gz")
    dfs.append(mat_df.assign(material=k))

print("Write to disk..")
pd.concat(dfs, ignore_index=True).to_pickle(PROC_SOURCE / f"ten_materials.pkl.gz")

print("Done.")