import pandas as pd
import numpy as np
import xgboost
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
from pprint import pprint
from utils.experiments import get_stratified_fold_indices, PROC_SOURCE, form_factor, crest_factor
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)


def main():
    ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    # drop H curve, we only take power loss as target
    ds = ds.drop(columns=[c for c in ds if c.startswith("H_t")])
    exp_log = {}
    feature_imp_sum= np.zeros(10)
    for material_lbl, mat_df in tqdm(
        ds.groupby("material"), desc="Train across materials"
    ):
        full_b = mat_df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()
        dbdt = full_b[:, 1:] - full_b[:, :-1]
        mat_df = mat_df.reset_index(drop=True)
        kfold_lbls = get_stratified_fold_indices(mat_df, 4)
        mat_df_proc = mat_df.assign(
            kfold=kfold_lbls,
            log_freq= np.log10(mat_df.loc[:,'freq']),
            #b_fft=np.fft.fft(full_b),
            b_fft_mean=np.mean(np.abs(np.fft.fft(full_b)),axis=1),
            b_peak2peak=full_b.max(axis=1) - full_b.min(axis=1),
            log_peak2peak = np.log10(full_b.max(axis=1) - full_b.min(axis=1)),
            max_dbdt=np.max(dbdt, axis=1),
            min_dbdt=np.min(dbdt, axis=1),
            mean_abs_dbdt=np.mean(np.abs(dbdt), axis=1),
            #crest_fac=crest_factor(full_b),
            form_fac=form_factor(full_b)
            # median_dbdt=np.median(dbdt, axis=1)
            # more features imaginable (count of spikes e.g.)
        ).drop(
            columns=[c for c in mat_df if c.startswith("B_t_")] + ["material"]
        )  # drop B curve
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
            y = train_fold_df.pop("ploss")
            X = train_fold_df.loc[:, x_cols]

            gbm = xgboost.XGBRegressor(max_depth=10, gamma = 0.05822,learning_rate=0.07, n_estimators=1100, subsample=0.89965, colsample_bytree=0.76261, objective='reg:squarederror')
            gbm.fit(X, y)
            pred = gbm.predict(test_fold_df.loc[:, x_cols])
            results_df.loc[results_df.kfold == kfold_lbl, "pred"] = pred
            feature_imp_sum += gbm.feature_importances_
            # plot

        # book keeping
        # print(feature_imp_sum)
        # plot
        

        exp_log[material_lbl] = calculate_metrics(
            results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
        )
    print("Overall Score")
    print(pd.DataFrame(exp_log).T)
    print(np.mean(pd.DataFrame(exp_log).T.loc[:,"avg-abs-rel-err"].to_numpy()))
    plt.bar(range(len(feature_imp_sum)), feature_imp_sum)
    plt.show()



if __name__ == "__main__":
    main()
