import os.path

from utils.experiments import get_waveform_est, BSAT_MAP
from run_cnn import B_COLS, H_COLS, construct_tensor_seq2seq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from utils.visualization import visualize_rel_error_report
from utils import metrics
from utils.experiments import get_bh_integral_from_two_mats
import shutil
import os

DATA_SOURCE = Path.cwd().parent / 'data' / 'input' / 'raw'
VAL_SOURCE = DATA_SOURCE.parent / 'validation'
PROC_SOURCE = DATA_SOURCE.parent/ "processed"
PREDS_SINK = PROC_SOURCE.parent.parent / 'output'
MODELS_SINK = PREDS_SINK.parent / 'models' / 'kfold_1_subsamples_4_cherrypicking'
#MODELS_SINK = PREDS_SINK.parent / 'models' / 'kfold_4_subsamples_4_cherrypicking'

def local_average(preds, gtruth):
    """Render a visual report as requested by the MagNet Challenge hosts, see
    (https://github.com/minjiechen/magnetchallenge/blob/main/pretest/PretestResultsPDF.pdf).
    Argument preds are the model predictions, gtruth the ground truth of the power losses.
    Both arguments must be either dictionaries or pandas DataFrames, in order to
    distinguish between materials. Moreover, both should be likewise sorted."""

    if isinstance(preds, dict):
        preds = pd.DataFrame(preds)
    if isinstance(gtruth, dict):
        gtruth = pd.DataFrame(gtruth)
    assert "material" in preds, 'Column "material" missing in preds argument'
    assert "material" in gtruth, 'Column "material" missing in gtruth argument'
    n_materials = gtruth.groupby("material").ngroups
    fig, axes = plt.subplots(
        nrows=np.ceil(n_materials / 2).astype(int),
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(8.26, 11.69),
    )
    # joined_df = pd.concat([gtruth.reset_index(drop=True), preds.reset_index(drop=True).drop(columns=['material'])], axis=1)
    for (m_lbl, preds_mat_df), ax in zip(preds.groupby("material"), axes.flatten()):
        preds_mat = preds_mat_df.loc[
            :, [c for c in preds_mat_df if c.startswith("h_pred")]
        ].to_numpy()
        gtruth_mat = gtruth.query("material == @m_lbl")

        p_pred = get_bh_integral_from_two_mats(
            freq=gtruth_mat.freq.to_numpy(),
            b=gtruth_mat.loc[
                :, [c for c in gtruth_mat if c.startswith("B_t_")]
            ].to_numpy()[:, :: 1024 // preds_mat.shape[1]],
            h=preds_mat,
        )

        assert (
            p_pred.shape[0] == gtruth_mat.ploss.shape[0]
        ), f"shapes mismatch, preds {m_lbl} != gtruth {gtruth.loc[gtruth.material == str(m_lbl), :].material_name.unique()[0]}, with {p_pred.shape=} != {gtruth_mat.shape=}"
        err = (np.abs(p_pred - gtruth_mat.ploss) / gtruth_mat.ploss * 100).ravel()
        avg = err.mean()
    return avg


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
    mat_df.to_pickle(PROC_SOURCE / f"{k}.pkl.gz")
    dfs.append(mat_df.assign(material=k))

val_ds = pd.concat(dfs, ignore_index=True)

result_dict_score = {}
result_dict_ds = {}
result_dict_h_pred_val = {}
result_dict_model_name = {}

for count_model, model in enumerate(MODELS_SINK.glob("*.pt")):
    mdl = torch.jit.load(model)
    mdl.eval()

    # extract the material name out of the model name
    material_name = str(model).replace('cnn_', '').replace(str(MODELS_SINK), '').replace('/', '').split('_')[0]

    print(f"{material_name = }")

    # here, model is trained on 3C90
    ds = val_ds.query(f"material == '{material_name}'").reset_index(drop=True)
    waveforms = get_waveform_est(
        ds.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()
    )
    ds = pd.concat(
        [
            ds,
            pd.get_dummies(waveforms, prefix="wav", dtype=float).rename(
                columns={
                    "wav_0": "wav_other",
                    "wav_1": "wav_square",
                    "wav_2": "wav_triangular",
                    "wav_3": "wav_sine",
                }
            ),
        ],
        axis=1,
    )

    full_b = ds.loc[:, B_COLS].to_numpy()
    dbdt = full_b[:, 1:] - full_b[:, :-1]
    b_peak2peak = full_b.max(axis=1) - full_b.min(axis=1)
    ds = ds.assign(
        b_peak2peak=b_peak2peak,
        log_peak2peak=np.log(b_peak2peak),
        mean_abs_dbdt=np.mean(np.abs(dbdt), axis=1),
        log_mean_abs_dbdt=np.log(np.mean(np.abs(dbdt), axis=1)),
        db_bsat=b_peak2peak / ds.material.map(BSAT_MAP),
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
    b_limit_per_profile = (
        np.abs(ds.loc[:, B_COLS].to_numpy())
        .max(axis=1)
        .reshape(-1, 1)
    )
    h_limit = h_limit * b_limit_per_profile / b_limit

    b_limit_test_fold = b_limit
    b_limit_test_fold_pp = b_limit_per_profile
    h_limit_test_fold = h_limit

    val_tensor_ts, val_tensor_scalar = construct_tensor_seq2seq(
        ds,
        x_cols,
        b_limit_test_fold,
        h_limit_test_fold,
        b_limit_pp=b_limit_test_fold_pp,
    )
    # val_tensor_ts = val_tensor_ts.to(device)
    # val_tensor_scalar = val_tensor_scalar.to(device)
    loss = torch.nn.MSELoss()
    mdl.eval()
    with torch.no_grad():
        val_pred = mdl(
            val_tensor_ts[:, :, :-1].permute(1, 2, 0),
            val_tensor_scalar,
        ).permute(2, 0, 1)
        val_g_truth = val_tensor_ts[:, :, [-1]]
        val_loss = loss(val_pred, val_g_truth).cpu().item()
        print(f"Loss: {val_loss:.2f}")
        val_pred = val_pred.squeeze().cpu().numpy().T * h_limit_test_fold
        h_pred_val = pd.DataFrame(val_pred,
                                  columns=[f"h_pred_{i}" for i in range(val_pred.shape[1])]).assign(material=material_name)

        # if count_model == 0:
        #     ds_merged = ds.copy()
        #     h_pred_val_merged = h_pred_val.copy()
        # else:
        #     ds_merged = pd.concat([ds_merged, ds], ignore_index=True)
        #     h_pred_val_merged = pd.concat([h_pred_val_merged, h_pred_val], ignore_index=True)


        mean_avg = local_average(h_pred_val, ds)

        if material_name not in result_dict_score.keys():
            print(f"{material_name}: first score")
            result_dict_score[material_name] = mean_avg
            result_dict_ds[material_name] = ds.copy()
            result_dict_h_pred_val[material_name] = h_pred_val.copy()
            result_dict_model_name[material_name] = model
        elif mean_avg < result_dict_score[material_name]:
            print(f"{material_name}: Current score {mean_avg} < stored score {result_dict_score[material_name]}. Overwrite score!")
            result_dict_score[material_name] = mean_avg
            result_dict_ds[material_name] = ds.copy()
            result_dict_h_pred_val[material_name] = h_pred_val.copy()
            result_dict_model_name[material_name] = model
        else:
            print(f"{material_name}: Current score {mean_avg} > stored score {result_dict_score[material_name]}. Skip.")


for count_keys, material_name_key in enumerate(result_dict_ds.keys()):

    if count_keys == 0:
        ds_merged = result_dict_ds[material_name_key].copy()
        h_pred_val_merged = result_dict_h_pred_val[material_name_key].copy()
    else:
        ds_merged = pd.concat([ds_merged, result_dict_ds[material_name_key].copy()], ignore_index=True)
        h_pred_val_merged = pd.concat([h_pred_val_merged, result_dict_h_pred_val[material_name_key].copy()], ignore_index=True)

# copy best models in sub-direction
best_model_path = os.path.join(MODELS_SINK, 'best_models')
if not os.path.exists(best_model_path):
    os.mkdir(best_model_path)

for material_name_key in result_dict_model_name.keys():
    model_path = result_dict_model_name[material_name_key]
    print(f"{model_path = }")
    shutil.copy(model_path, best_model_path)


fig = visualize_rel_error_report(h_pred_val_merged, ds_merged)
fig.savefig(MODELS_SINK / "best_models" / f"error_report.pdf", dpi=300, bbox_inches='tight')