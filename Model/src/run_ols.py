"""Run linear regression with regularization training"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pprint import pprint
from utils.experiments import (
    get_stratified_fold_indices,
    PROC_SOURCE,
    get_waveform_est,
    get_bh_integral_from_two_mats,
)
from utils.metrics import calculate_metrics

pd.set_option("display.max_columns", None)

INCLUDE_H_PRED = (
    PROC_SOURCE.parent.parent
    / "output"
    / "CNN_H_preds_16-Sep-2023_06:55_Uhr_score_2.86.csv.zip"
)


def main():
    ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    # drop H curve, we only take power loss as target
    ds = ds.drop(columns=[c for c in ds if c.startswith("H_t")])
    if INCLUDE_H_PRED is not None:
        h_preds = pd.read_csv(INCLUDE_H_PRED, dtype={"material": str})
        orig_p_score_d = {}
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
    exp_log = {}
    for material_lbl, mat_df in tqdm(
        ds.groupby("material"), desc="Train across materials"
    ):
        full_b = mat_df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()
        dbdt = full_b[:, 1:] - full_b[:, :-1]
        mat_df = mat_df.reset_index(drop=True)
        kfold_lbls = get_stratified_fold_indices(mat_df, 4)

        mat_df_proc = mat_df.assign(
            kfold=kfold_lbls,
            b_peak2peak=full_b.max(axis=1) - full_b.min(axis=1),
            log_b_peak2peak=np.log(full_b.max(axis=1) - full_b.min(axis=1)),
            # max_dbdt=np.max(dbdt, axis=1),
            # min_dbdt=np.min(dbdt, axis=1),
            log_mean_abs_dbdt=np.log(np.mean(np.abs(dbdt), axis=1)),
            # median_dbdt=np.median(dbdt, axis=1),
            log_freq=np.log(mat_df.freq),
            # more features imaginable (count of spikes e.g.)
        ).drop(
            columns=[c for c in mat_df if c.startswith("B_t_")] + ["material"]
        )  # drop B curve

        if INCLUDE_H_PRED is not None:
            full_h = (
                h_preds.query("material == @material_lbl")
                .reset_index(drop=True)
                .drop(columns=["material"])
                .to_numpy()
            )
            dhdt = full_h[:, 1:] - full_h[:, :-1]
            p_derived_from_h = get_bh_integral_from_two_mats(
                mat_df_proc.freq, full_b[:, :: 1024 // full_h.shape[1]], full_h
            )
            log_p_derived_from_h = np.log(p_derived_from_h)
            mat_df_proc = mat_df_proc.assign(
                # h_peak2peak=full_h.max(axis=1) - full_h.min(axis=1),
                log_h_peak2peak=np.log(full_h.max(axis=1) - full_h.min(axis=1)),
                # max_dhdt=np.max(dhdt, axis=1),
                # min_dhdt=np.min(dhdt, axis=1),
                log_mean_abs_dhdt=np.log(np.mean(np.abs(dhdt), axis=1)),
                p_derived_from_h=p_derived_from_h,
                log_p_derived_from_h=log_p_derived_from_h,
            )
            orig_p_score_d[material_lbl] = calculate_metrics(
                p_derived_from_h, mat_df_proc.ploss
            )

        # training result container
        results_df = mat_df_proc.loc[:, ["ploss", "kfold"]].assign(pred=0)
        x_cols = [c for c in mat_df_proc if c not in ["ploss", "kfold"]]
        print(x_cols)
        for kfold_lbl, test_fold_df in mat_df_proc.groupby("kfold"):
            train_fold_df = (
                mat_df_proc.query("kfold != @kfold_lbl")
                .reset_index(drop=True)
                .drop(columns="kfold")
            )
            assert len(train_fold_df) > 0, "empty dataframe error"
            y = train_fold_df.pop("ploss").to_numpy()
            
            X = train_fold_df.loc[:, x_cols].to_numpy()
            # normalize
            x_scaler = StandardScaler()
            #X_max = np.abs(X).max(axis=0).reshape(1, -1)
            X = x_scaler.fit_transform(X)
            y = np.log(y)
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1))
            #y_max = y.max()
            mdl = Ridge(alpha=1e-6) #LinearRegression()
            mdl.fit(X, y)
            pred = mdl.predict(x_scaler.transform(test_fold_df.loc[:, x_cols].to_numpy()))
            results_df.loc[results_df.kfold == kfold_lbl, "pred"] = (
                np.exp(y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel())
            )  # denormalize

        # book keeping
        exp_log[material_lbl] = calculate_metrics(
            results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
        )
    print("Overall Score")
    if INCLUDE_H_PRED:
        exp_result_df = pd.concat(
                [
                    pd.DataFrame(exp_log)
                    .T.loc[:, ["avg-abs-rel-err"]]
                    .rename(columns={"avg-abs-rel-err": "postprocessing_score"}),
                    pd.DataFrame(orig_p_score_d)
                    .T.loc[:, ["avg-abs-rel-err"]]
                    .rename(columns={"avg-abs-rel-err": "orig_p_score"}),
                ],
                axis=1,
            )
        print(exp_result_df)
        print("Mean:")
        print(exp_result_df.mean(axis=0))
    else:
        print(pd.DataFrame(exp_log).T.loc[:, ["avg-abs-rel-err"]])


if __name__ == "__main__":
    main()
