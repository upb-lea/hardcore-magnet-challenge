"""Data and file handling"""
from pathlib import Path
import pandas as pd
from joblib import delayed, Parallel

NEW_MATS_ROOT_PATH = Path.cwd().parent / "data" / "input" / "test" / "Training"
ALL_B_COLS = [f"B_t_{k}" for k in range(1024)]
ALL_H_COLS = [f"H_t_{k}" for k in range(1024)]


def load_material_csv_files_and_generate_pandas_df(mat_folder_path):
    mat_lbl = mat_folder_path.name.split(' ')[-1]
    b_df = pd.read_csv(mat_folder_path / "B_Field.csv", header=0, names=ALL_B_COLS)
    h_df = pd.read_csv(mat_folder_path / "H_Field.csv", header=0, names=ALL_H_COLS)
    freq = pd.read_csv(mat_folder_path / 'Frequency.csv', header=0)
    temp = pd.read_csv(mat_folder_path / "Temperature.csv", header=0)
    ploss = pd.read_csv(mat_folder_path / "Volumetric_Loss.csv", header=0)

    return pd.concat([b_df, h_df], axis=1).assign(freq=freq, temp=temp, ploss=ploss, material=mat_lbl)

def load_new_materials_for_training():

    with Parallel(n_jobs=5) as prll:
        mats_l = prll(delayed(load_material_csv_files_and_generate_pandas_df)(mat_folder) for mat_folder in NEW_MATS_ROOT_PATH.glob("Material*"))
    return pd.concat(mats_l, axis=0, ignore_index=True)


        
