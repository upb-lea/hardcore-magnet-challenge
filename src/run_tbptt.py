"""Run truncated backprop through time (TBPTT) training [seq2seq].
In order to mitigate a skewed initial condition (hidden state) not only
one period is estimated but several. The last period will be cropped off
the full prediction sequence and used for evaluation """
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from joblib import delayed, Parallel
from pprint import pprint
import torch
from torchinfo import summary as ti_summary
import random
from torchdiffeq import odeint_adjoint, odeint
from utils.experiments import (
    get_stratified_fold_indices,
    PROC_SOURCE,
    get_bh_integral_from_two_mats,
)
from utils.metrics import calculate_metrics
from utils.topology import DifferenceEqLayer, ExplEulerCell


pd.set_option("display.max_columns", None)

DEBUG = False
N_SEEDS = 2  # how often should the experiment be repeated with different random init
N_JOBS = -1  # how many processes should be working
N_EPOCHS = 3 if DEBUG else 100  # how often should the full data set be iterated over
half_lr_at = [80, 90]  # halve learning rate after these many epochs
SUBSAMPLE_FACTOR = 8  # every n-th sample along the time axis is considered
K_KFOLD = 4  # how many folds in cross validation
BATCH_SIZE = 1024  # how many periods/profiles/measurements should be averaged across for a weight update
TBPTT_LEN = (
    1024 // SUBSAMPLE_FACTOR
)  # after how many prediction steps do we have a weight update
N_PERIODS = 3  # how many periods should be predicted

B_COLS = [f"B_t_{k}" for k in range(0, 1024, SUBSAMPLE_FACTOR)]
H_COLS = [f"H_t_{k}" for k in range(0, 1024, SUBSAMPLE_FACTOR)]


def construct_tensor_seq2seq(df, x_cols, b_limit, h_limit):
    full_b = df.loc[:, B_COLS].to_numpy()
    full_h = df.loc[:, H_COLS].to_numpy()
    df = df.drop(columns=[c for c in df if c.startswith(("H_t_", "B_t_"))])
    assert len(df) > 0, "empty dataframe error"
    # put freq on first place since Architecture expects it there
    x_cols.insert(0, x_cols.pop(x_cols.index("freq")))
    X = df.loc[:, x_cols]
    # normalization
    full_b /= b_limit
    full_h /= h_limit
    X.loc[:, ["temp", "freq"]] /= np.array([75.0, 500_000])
    other_cols = [c for c in x_cols if c not in ["temp", "freq"]]
    X.loc[:, other_cols] /= X.loc[:, other_cols].abs().max(axis=0)
    # generate tensors with shape: (#time steps, #profiles/periods, #features)
    tens = torch.dstack(
        [
            torch.tensor(
                np.repeat(X.to_numpy()[np.newaxis, ...], full_b.shape[1], axis=0),
                dtype=torch.float32,
            ),
            torch.tensor(full_b.T[..., np.newaxis], dtype=torch.float32),
            torch.tensor(
                full_h.T[..., np.newaxis], dtype=torch.float32
            ),  # target is last column
        ]
    )
    return tens


def main():
    device = torch.device("cpu")
    ds = pd.read_pickle(
        PROC_SOURCE / "3C90.pkl.gz" if DEBUG else PROC_SOURCE / "ten_materials.pkl.gz"
    )

    
    for material_lbl, mat_df in (("3C90", ds),) if DEBUG else ds.groupby("material"):
        mat_df = mat_df.reset_index(drop=True)

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
                "models_state_dict": [],
                "start_time": pd.Timestamp.now().round(freq="S"),
            }
            # training result container
            results_df = mat_df_proc.loc[:, ["ploss", "kfold"]].assign(pred=0)
            x_cols = [
                c
                for c in mat_df_proc
                if c not in ["ploss", "kfold"] and not c.startswith(("B_t_", "H_t_"))
            ]
            # store max elongation for normalization
            b_limit = np.abs(mat_df_proc.loc[:, B_COLS].to_numpy()).max()
            h_limit = np.abs(mat_df_proc.loc[:, H_COLS].to_numpy()).max()
            for kfold_lbl, test_fold_df in mat_df_proc.groupby("kfold"):
                train_fold_df = (
                    mat_df_proc.query("kfold != @kfold_lbl")
                    .reset_index(drop=True)
                    .drop(columns="kfold")
                )
                train_tensor = construct_tensor_seq2seq(
                    train_fold_df, x_cols, b_limit, h_limit
                )
                # mirror time axis, since H field seems to be more causally caused by B field in reverse
                train_tensor = torch.flip(train_tensor, dims=(0,))

                mdl = DifferenceEqLayer(
                    ExplEulerCell,
                    n_targets=1,
                    n_input_feats=len(x_cols) + 1,  # +b curve
                    subsample_factor=SUBSAMPLE_FACTOR,
                    layer_cfg=None,
                ).to(device)
                mdl = torch.jit.script(mdl)
                opt = torch.optim.NAdam(mdl.parameters(), lr=1e-3)
                loss = torch.nn.MSELoss().to(device)
                pbar = trange(
                    N_EPOCHS,
                    desc=f"Seed {rep}, fold {kfold_lbl}",
                    position=rep * K_KFOLD + kfold_lbl,
                    unit="epoch",
                )
                if rep == 0 and kfold_lbl == 0:  # print only once
                    info_kwargs = {}
                    info_kwargs["state"] = torch.randn(
                        (1, 1), dtype=torch.float32, device=device
                    )
                    mdl_info = ti_summary(
                        mdl,
                        input_size=(1, 1, len(x_cols) + 1),
                        verbose=0,
                        **info_kwargs,
                    )
                    pbar.write(str(mdl_info))
                    logs["model_size"] = mdl_info.total_params
                mdl.to(device)

                # generate shuffled indices beforehand
                n_profiles = train_tensor.shape[1]
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
                    train_tensor_shuffled = train_tensor[:, indices, :]
                    n_profiles = train_tensor_shuffled.shape[1]
                    # randomly shift time axis (all profiles the same amount)
                    train_tensor_shuffled = torch.roll(
                        train_tensor_shuffled,
                        shifts=np.random.randint(0, train_tensor.shape[0] - 1),
                        dims=0,
                    )
                    for i_batch in range(int(np.ceil(n_profiles / BATCH_SIZE))):
                        # extract mini-batch
                        start_marker = i_batch * BATCH_SIZE
                        end_marker = min((i_batch + 1) * BATCH_SIZE, n_profiles)
                        train_tensor_shuffled_n_batched = train_tensor_shuffled[
                            :, start_marker:end_marker, :
                        ]
                        hidden = torch.zeros(
                            (train_tensor_shuffled_n_batched.shape[1], 1),
                            dtype=train_tensor_shuffled_n_batched.dtype,
                        )  # only one target (H curve)

                        # repeat profile couple of times to estimate over
                        train_tensor_shuffled_n_batched = (
                            train_tensor_shuffled_n_batched.repeat((N_PERIODS, 1, 1))
                        )
                        n_seqs = int(
                            np.ceil(
                                train_tensor_shuffled_n_batched.shape[0] / TBPTT_LEN
                            )
                        )
                        for i_seq in range(n_seqs):
                            # iteration from beginning to end of subsequences to average gradients across
                            mdl.zero_grad()
                            hidden = hidden.detach()
                            train_sample = train_tensor_shuffled_n_batched[
                                i_seq * TBPTT_LEN : (i_seq + 1) * TBPTT_LEN, :, :
                            ]
                            g_truth = train_sample[:, :, [-1]]
                            X_tensor = train_sample[:, :, :-1]
                            output, hidden = mdl(X_tensor, hidden)
                            if i_seq > 0:
                                # skip first period since initial condition cannot be correct
                                train_loss = loss(output, g_truth)
                                train_loss.backward()
                                opt.step()
                    with torch.no_grad():
                        logs["loss_trends_train"][kfold_lbl].append(train_loss.item())
                        pbar_str = f"Loss {train_loss.item():.2e}"
                    # validation set
                    val_tensor = construct_tensor_seq2seq(
                        test_fold_df, x_cols, b_limit, h_limit
                    )
                    val_tensor = torch.flip(val_tensor, dims=(0,))
                    with torch.no_grad():
                        mdl.eval()
                        hidden = torch.zeros(
                            (val_tensor.shape[1], 1),
                            dtype=val_tensor.dtype,
                        )  # only one target (H curve)

                        val_pred, hidden = mdl(val_tensor[:, :, :-1], hidden)
                        val_pred_cropped = val_pred[-1024 // SUBSAMPLE_FACTOR :, :, :]
                        val_tensor_cropped = val_tensor[
                            -1024 // SUBSAMPLE_FACTOR :, :, :
                        ]
                        val_gtruth_cropped = val_tensor_cropped[:, :, [-1]]
                        val_loss = loss(val_pred_cropped, val_gtruth_cropped).item()

                        logs["loss_trends_val"][kfold_lbl].append(val_loss)
                        pbar_str += f"| val loss {val_loss:.2e}"
                    pbar.set_postfix_str(pbar_str)
                    if np.isnan(val_loss):
                        break
                    if half_lr_at is not None:
                        if i_epoch in half_lr_at:
                            for group in opt.param_groups:
                                group["lr"] *= 0.5
                    with torch.inference_mode():
                        results_df.loc[
                            results_df.kfold == kfold_lbl, "pred"
                        ] = get_bh_integral_from_two_mats(
                            freq=val_tensor_cropped[0, :, 0].numpy().reshape(-1, 1),
                            b=val_tensor_cropped[:, :, -2].numpy().T * b_limit,
                            h=val_pred_cropped.squeeze().numpy().T * h_limit,
                        )

            # book keeping
            logs[material_lbl] = calculate_metrics(
                results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
            )
            return logs

        n_seeds = 1 if DEBUG else N_SEEDS
        print(f"Parallelize over {n_seeds} seeds with {N_JOBS} processes..")
        # start experiments in parallel processes
        with Parallel(n_jobs=N_JOBS) as prll:
            # list of dicts
            experiment_logs = prll(delayed(run_dyn_training)(s) for s in range(n_seeds))
    print("Overall Score")
    for i, logs in enumerate(experiment_logs):
        print("Seed", i)
        mat_lbls = ['3C90'] if DEBUG else ds.material.unique().tolist()
        print(pd.DataFrame({k: logs[k] for k in mat_lbls if k in logs}).T)


if __name__ == "__main__":
    main()
