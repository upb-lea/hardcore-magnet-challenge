import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
from pprint import pprint
from utils.experiments import get_stratified_fold_indices, PROC_SOURCE
from utils.metrics import calculate_metrics

pd.set_option("display.max_columns", None)


def main():
    ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    # drop H curve, we only take power loss as target
    ds = ds.drop(columns=[c for c in ds if c.startswith("H_t")])
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
            max_dbdt=np.max(dbdt, axis=1),
            min_dbdt=np.min(dbdt, axis=1),
            mean_dbdt=np.mean(dbdt, axis=1),
            median_dbdt=np.median(dbdt, axis=1)
            # more features imaginable (count of spikes e.g.)
        ).drop(
            columns=[c for c in mat_df if c.startswith("B_t_")] + ["material"]
        )  # drop B curve

        # training result container
        results_df = mat_df_proc.loc[:, ["ploss", "kfold"]].assign(pred=0)

        for kfold_lbl, test_fold_df in mat_df_proc.groupby("kfold"):
            train_fold_df = mat_df_proc.query("kfold == @kfold_lbl").reset_index(
                drop=True
            )
            assert len(train_fold_df) > 0, "empty dataframe error"
            y = train_fold_df.pop("ploss")
            X = train_fold_df

            mdl = Ridge()  # LinearRegression()
            mdl.fit(X.to_numpy(), y.to_numpy())
            pred = mdl.predict(
                test_fold_df.loc[
                    :, [c for c in test_fold_df if c != "ploss"]
                ].to_numpy()
            )
            results_df.loc[results_df.kfold == kfold_lbl, "pred"] = pred

        # book keeping
        exp_log[material_lbl] = calculate_metrics(
            results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
        )
    print("Overall Score")
    print(pd.DataFrame(exp_log).T)


if __name__ == "__main__":
    main()
