"""Run convolutional neural networks """
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from joblib import delayed, Parallel
from pprint import pprint
import torch
from torchinfo import summary as ti_summary
import random
from datetime import datetime

from utils.experiments import (
    get_stratified_fold_indices,
    PROC_SOURCE,
    MODEL_SINK,
    BSAT_MAP,
    PRED_SINK,
    get_bh_integral_from_two_mats,
    get_waveform_est,
)
from utils.metrics import calculate_metrics
from utils.topology import TemporalAcausalConvNet, TCNWithScalarsAsBias


pd.set_option("display.max_columns", None)

DEBUG = True
N_SEEDS = 3  # how often should the experiment be repeated with different random init
N_JOBS = 1  # how many processes should be working
N_EPOCHS = 5 if DEBUG else 2000  # how often should the full data set be iterated over
half_lr_at = [int(N_EPOCHS * 0.8)]  # halve learning rate after these many epochs
SUBSAMPLE_FACTOR = 4  # every n-th sample along the time axis is considered
FREQ_SCALE = 150_000  # in Hz
K_KFOLD = 1 if DEBUG else 4  # how many folds in cross validation
BATCH_SIZE = 64  # how many periods/profiles/measurements should be averaged across for a weight update


B_COLS = [f"B_t_{k}" for k in range(0, 1024, SUBSAMPLE_FACTOR)]
H_COLS = [f"H_t_{k}" for k in range(0, 1024, SUBSAMPLE_FACTOR)]
H_PRED_COLS = [f"h_pred_{i}" for i in range(1024 // SUBSAMPLE_FACTOR)]
DEBUG_MATERIALS = ["3C90", "78"]


def construct_tensor_seq2seq(df, x_cols, b_limit, h_limit, b_limit_pp=None):
    """generate tensors with shape: (#time steps, #profiles/periods, #features)"""
    full_b = df.loc[:, B_COLS].to_numpy()
    full_h = df.loc[:, H_COLS].to_numpy()
    df = df.drop(columns=[c for c in df if c.startswith(("H_t_", "B_t_", "material"))])
    assert len(df) > 0, "empty dataframe error"
    # put freq on first place since Architecture expects it there
    x_cols.insert(0, x_cols.pop(x_cols.index("freq")))
    X = df.loc[:, x_cols]
    # normalization
    full_b /= b_limit
    full_h /= h_limit
    X.loc[:, ["temp", "freq"]] /= np.array([75.0, FREQ_SCALE])
    X.loc[:, "freq"] = np.log(X.freq)
    other_cols = [c for c in x_cols if c not in ["temp", "freq"]]
    X.loc[:, other_cols] /= X.loc[:, other_cols].abs().max(axis=0)
    # tensor list
    tens_l = []
    if b_limit_pp is not None:
        # add another B curve with different normalization
        per_profile_scaled_b = full_b * b_limit / b_limit_pp
        freq = X.loc[:, "freq"].to_numpy().reshape(-1, 1)
        # get derivatives
        b_deriv = np.empty((full_b.shape[0], full_b.shape[1] + 2))
        b_deriv[:, 1:-1] = per_profile_scaled_b
        b_deriv[:, 0] = per_profile_scaled_b[:, -1]
        b_deriv[:, -1] = per_profile_scaled_b[:, 0]
        b_deriv = np.gradient(b_deriv, axis=1) * freq
        b_deriv_sq = np.gradient(b_deriv, axis=1) * freq
        b_deriv = b_deriv[:, 1:-1]
        b_deriv_sq = b_deriv_sq[:, 1:-1]
        tantan_b = -np.tan(0.9 * np.tan(per_profile_scaled_b)) / 6  # tan-tan feature
        tens_l += [
            torch.tensor(per_profile_scaled_b.T[..., np.newaxis], dtype=torch.float32),
            torch.tensor(
                b_deriv.T[..., np.newaxis] / np.abs(b_deriv).max(), dtype=torch.float32
            ),
            torch.tensor(
                b_deriv_sq.T[..., np.newaxis] / np.abs(b_deriv_sq).max(),
                dtype=torch.float32,
            ),
            torch.tensor(tantan_b.T[..., np.newaxis], dtype=torch.float32),
        ]
    tens_l += [
        torch.tensor(
            full_b.T[..., np.newaxis], dtype=torch.float32
        ),  # b field is penultimate column
        torch.tensor(
            full_h.T[..., np.newaxis], dtype=torch.float32
        ),  # target is last column
    ]

    # return ts tensor with shape: (#time steps, #profiles, #features), and scalar tensor with (#profiles, #features)
    return torch.dstack(tens_l), torch.tensor(X.to_numpy(), dtype=torch.float32)


def main(ds=None, start_seed=0, per_profile_norm=False):
    device = torch.device("cuda")
    if ds is None:
        ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    elif DEBUG:
        ds = ds.query("material in @DEBUG_MATERIALS")

    logs_d = {}

    for m_i, (material_lbl, mat_df) in enumerate(ds.groupby("material")):
        mat_df = mat_df.reset_index(drop=True)
        print(f"Train for {material_lbl}")

        mat_df_proc = mat_df.assign(
            kfold=get_stratified_fold_indices(mat_df, K_KFOLD),
        )
        if "material" in mat_df_proc:
            mat_df_proc = mat_df_proc.drop(columns=["material"])

        def run_dyn_training(rep=0):
            # seed
            np.random.seed(rep)
            random.seed(rep)
            torch.manual_seed(rep)

            logs = {
                "loss_trends_train": [[] for _ in range(K_KFOLD)],
                "loss_trends_val": [[] for _ in range(K_KFOLD)],
                "model_scripted": [],
                "start_time": pd.Timestamp.now().round(freq="S"),
                "performance": None,
            }
            # training result container
            results_df = mat_df_proc.loc[:, ["ploss", "kfold"]].assign(pred=0)
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        np.zeros((len(results_df), len(H_PRED_COLS))),
                        columns=H_PRED_COLS,
                    ),
                ],
                axis=1,
            )

            x_cols = [
                c
                for c in mat_df_proc
                if c not in ["ploss", "kfold"] and not c.startswith(("B_t_", "H_t_"))
            ]

            # store max elongation for normalization
            b_limit = np.abs(mat_df_proc.loc[:, B_COLS].to_numpy()).max()  # T
            h_limit = min(
                np.abs(mat_df_proc.loc[:, H_COLS].to_numpy()).max(), 150
            )  # A/m
            if per_profile_norm:
                # normalize on a per-profile base
                b_limit_per_profile = (
                    np.abs(mat_df_proc.loc[:, B_COLS].to_numpy())
                    .max(axis=1)
                    .reshape(-1, 1)
                )
                h_limit = h_limit * b_limit_per_profile / b_limit

            for kfold_lbl, test_fold_df in mat_df_proc.groupby("kfold"):
                if K_KFOLD > 1:
                    train_fold_df = mat_df_proc.query("kfold != @kfold_lbl")
                else:
                    train_fold_df = mat_df_proc
                train_idx = train_fold_df.index.to_numpy()
                train_fold_df = train_fold_df.reset_index(drop=True).drop(
                    columns="kfold"
                )

                if per_profile_norm:
                    b_limit_fold = b_limit
                    b_limit_fold_pp = b_limit_per_profile[train_idx]
                    h_limit_fold = h_limit[train_idx]
                else:
                    b_limit_fold = b_limit
                    b_limit_fold_pp = None
                    h_limit_fold = h_limit
                train_tensor_ts, train_tensor_scalar = construct_tensor_seq2seq(
                    train_fold_df,
                    x_cols,
                    b_limit_fold,
                    h_limit_fold,
                    b_limit_pp=b_limit_fold_pp,
                )
                train_tensor_ts = train_tensor_ts.to(device)
                train_tensor_scalar = train_tensor_scalar.to(device)

                n_ts = 4  # number of time series per profile next to B curve
                mdl = TCNWithScalarsAsBias(
                    num_input_scalars=len(x_cols),
                    num_input_ts=1 + int(per_profile_norm) * n_ts,
                    tcn_layer_cfg=None,
                    scalar_layer_cfg=None,
                )
                opt = torch.optim.NAdam(mdl.parameters(), lr=1e-3)
                loss = torch.nn.MSELoss().to(device)
                pbar = trange(
                    N_EPOCHS,
                    desc=f"Seed {rep}, fold {kfold_lbl}",
                    position=rep * K_KFOLD + kfold_lbl,
                    unit="epoch",
                    mininterval=1.0,
                )
                if rep == 0 and kfold_lbl == 0 and m_i == 0:  # print only once
                    mdl_info = ti_summary(
                        mdl,
                        input_data=[
                            torch.ones(
                                (1, 1 + int(per_profile_norm) * n_ts, len(H_COLS)),
                                dtype=torch.float32,
                            ),
                            torch.ones((1, len(x_cols)), dtype=torch.float32),
                        ],
                        device=device,
                        verbose=0,
                    )
                    pbar.write(str(mdl_info))
                    logs["model_size"] = mdl_info.total_params
                mdl.to(device)

                # generate shuffled indices beforehand
                n_profiles = train_tensor_ts.shape[1]
                idx_mat = []
                for _ in range(N_EPOCHS):
                    idx = np.arange(n_profiles)
                    np.random.shuffle(idx)
                    idx_mat.append(idx)
                idx_mat = np.vstack(idx_mat)

                # Training loop
                for i_epoch in pbar:
                    mdl.train()
                    # shuffle profiles
                    indices = idx_mat[i_epoch]
                    train_tensor_ts_shuffled = train_tensor_ts[:, indices, :]
                    train_tensor_scalar_shuffled = train_tensor_scalar[indices, :]
                    val_loss = None
                    # randomly shift time axis (all profiles the same amount)
                    #  Not necessary for CNNs!
                    # train_tensor_shuffled = torch.roll(
                    #    train_tensor_shuffled,
                    #    shifts=np.random.randint(0, train_tensor.shape[0] - 1),
                    #    dims=0,
                    # )
                    for i_batch in range(int(np.ceil(n_profiles / BATCH_SIZE))):
                        # extract mini-batch
                        start_marker = i_batch * BATCH_SIZE
                        end_marker = min((i_batch + 1) * BATCH_SIZE, n_profiles)
                        train_tensor_ts_shuffled_n_batched = train_tensor_ts_shuffled[
                            :, start_marker:end_marker, :
                        ]
                        train_tensor_scalar_shuffled_n_batched = (
                            train_tensor_scalar_shuffled[start_marker:end_marker, :]
                        )

                        # iteration from beginning to end of subsequences to average gradients across
                        mdl.zero_grad()

                        g_truth = train_tensor_ts_shuffled_n_batched[:, :, [-1]]
                        X_tensor_ts = train_tensor_ts_shuffled_n_batched[:, :, :-1]
                        X_tensor_scalar = train_tensor_scalar_shuffled_n_batched
                        output = mdl(
                            X_tensor_ts.permute(1, 2, 0), X_tensor_scalar
                        ).permute(2, 0, 1)

                        train_loss = loss(output, g_truth)
                        train_loss.backward()
                        opt.step()
                    with torch.no_grad():
                        logs["loss_trends_train"][kfold_lbl].append(
                            train_loss.cpu().item()
                        )
                        pbar_str = f"Loss {train_loss.cpu().item():.2e}"
                    
                    if K_KFOLD > 1:
                        do_validate = i_epoch % 10 == 0 or i_epoch == N_EPOCHS - 1
                    else:
                        do_validate = i_epoch == N_EPOCHS - 1
                    if do_validate:
                        # validation set
                        if per_profile_norm:
                            test_idx = test_fold_df.index.to_numpy()
                            b_limit_test_fold = b_limit
                            b_limit_test_fold_pp = b_limit_per_profile[test_idx]
                            h_limit_test_fold = h_limit[test_idx]
                        else:
                            b_limit_test_fold = b_limit
                            h_limit_test_fold = h_limit
                            b_limit_test_fold_pp = None
                        val_tensor_ts, val_tensor_scalar = construct_tensor_seq2seq(
                            test_fold_df,
                            x_cols,
                            b_limit_test_fold,
                            h_limit_test_fold,
                            b_limit_pp=b_limit_test_fold_pp,
                        )
                        val_tensor_ts = val_tensor_ts.to(device)
                        val_tensor_scalar = val_tensor_scalar.to(device)

                        mdl.eval()
                        with torch.no_grad():
                            val_pred = mdl(
                                val_tensor_ts[:, :, :-1].permute(1, 2, 0),
                                val_tensor_scalar,
                            ).permute(2, 0, 1)
                            val_g_truth = val_tensor_ts[:, :, [-1]]
                            val_loss = loss(val_pred, val_g_truth).cpu().item()
                            logs["loss_trends_val"][kfold_lbl].append(val_loss)
                        if np.isnan(val_loss):
                            break
                    if val_loss is not None:
                        pbar_str += f"| val loss {val_loss:.2e}"
                        
                    pbar.set_postfix_str(pbar_str)
                    
                    if half_lr_at is not None:
                        if i_epoch in half_lr_at:
                            for group in opt.param_groups:
                                group["lr"] *= 0.75
                    if i_epoch == N_EPOCHS - 1:  # last epoch
                        with torch.inference_mode():  # take last epoch's model as best model
                            val_tensor_ts_np = val_tensor_ts.cpu().numpy()
                            val_tensor_scalars_np = val_tensor_scalar.cpu().numpy()
                            h_pred_val = (
                                val_pred.squeeze().cpu().numpy().T * h_limit_test_fold
                            )
                            results_df.loc[
                                results_df.kfold == kfold_lbl, "pred"
                            ] = get_bh_integral_from_two_mats(
                                freq=np.exp(val_tensor_scalars_np[:, 0].reshape(-1, 1))
                                * FREQ_SCALE,
                                b=val_tensor_ts_np[:, :, -2].T * b_limit_test_fold,
                                h=h_pred_val,
                            )
                            results_df.loc[
                                results_df.kfold == kfold_lbl,
                                [c for c in results_df if c.startswith("h_pred_")],
                            ] = h_pred_val
                # end of fold
                logs["model_scripted"].append(torch.jit.script(mdl.cpu()))

            # book keeping
            logs["performance"] = calculate_metrics(
                results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
            )
            logs["results_df"] = results_df
            return logs

        n_seeds = N_SEEDS
        print(f"Parallelize over {n_seeds} seeds with {N_JOBS} processes..")
        # start experiments in parallel processes
        # list of dicts
        # mat_log = prll(delayed(run_dyn_training)(s) for s in range(start_seed, n_seeds + start_seed))
        logs_d[material_lbl] = [
            run_dyn_training(i) for i in range(start_seed, n_seeds + start_seed)
        ]
        # logs_d[material_lbl] = {'performance': pd.DataFrame.from_dict([m['performance'] for m in mat_log]),
        #                        'misc': [m for m in mat_log]}

    return logs_d


if __name__ == "__main__":
    # prepare folder structure for sinks
    PRED_SINK.mkdir(parents=True, exist_ok=True)
    MODEL_SINK.mkdir(parents=True, exist_ok=True)
    # load data set and featurize
    ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
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
    logs = main(ds=ds, per_profile_norm=True)
    print("Overall Score")
    performances_df = pd.DataFrame(
        {
            material: {
                f"seed_{i}": mm["performance"]["avg-abs-rel-err"]
                for i, mm in enumerate(seed_logs_l)
            }
            for material, seed_logs_l in logs.items()
        }
    )
    print(performances_df)  # (#seeds, #materials)
    best_seed = np.argmin(performances_df.to_numpy().mean(axis=1))
    best_score = np.min(performances_df.to_numpy().mean(axis=1))
    print(f"Mean Score: {best_score*100:.2f} %")

    # store predictions for post-processing
    print("Store predictions to disk..", end="")

    h_preds_df = pd.concat(
        [
            seed_logs_l[best_seed]["results_df"]
            .loc[:, H_PRED_COLS]
            .assign(material=material)
            for material, seed_logs_l in logs.items()
        ],
        ignore_index=True,
    )
    h_preds_df.to_csv(
        PRED_SINK
        / f"CNN_H_preds_{datetime.now().strftime('%d-%b-%Y_%H:%M_Uhr')}_score_{best_score*100:.2f}.csv.zip",
        index=False,
    )
    print("done.")

    # store jitted models
    print("Store models as jit-script to disk..", end="")
    for mat_lbl, seed_logs_l in logs.items():
        for seed_i, seed_log in enumerate(seed_logs_l):
            for fold_i, scripted_mdl in enumerate(seed_log["model_scripted"]):
                scripted_mdl.save(
                    MODEL_SINK
                    / (f"cnn_{mat_lbl}_{datetime.now().strftime('%d-%b-%Y_%H:%M_Uhr')}_"
                       f"score_{seed_log['performance']['avg-abs-rel-err']*100:.2f}_seed_{seed_i}_fold_{fold_i}.pt")
                )
    print("done.")