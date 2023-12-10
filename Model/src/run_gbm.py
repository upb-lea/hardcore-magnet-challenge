import pandas as pd
import numpy as np
import xgboost
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
from pprint import pprint
from utils.experiments import get_bh_integral_from_two_mats,get_stratified_fold_indices, PROC_SOURCE, form_factor, crest_factor
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid
from utils.metrics import calculate_metrics

pd.set_option("display.max_columns", None)

INCLUDE_H_PRED = (
    PROC_SOURCE.parent.parent
    / "output"
    / "CNN_pred.csv"
)

def main():
    ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    # drop H curve, we only take power loss as target
    ds = ds.drop(columns=[c for c in ds if c.startswith("H_t")])
    if INCLUDE_H_PRED is not None:
        h_preds = pd.read_csv(INCLUDE_H_PRED, dtype={"material": str})
        orig_p_score_d = {}
    exp_log = {}
    cols=[]
    feature_imp_sum= np.zeros(11)
    for material_lbl, mat_df in tqdm(
        ds.groupby("material"), desc="Train across materials"
    ):
        full_b = mat_df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()
        dbdt = full_b[:, 1:] - full_b[:, :-1]
        mat_df = mat_df.reset_index(drop=True)
        
        kfold_lbls = get_stratified_fold_indices(mat_df, 4)
        mat_df_proc = mat_df.assign(
            kfold=kfold_lbls,
            log_freq= np.log(mat_df.loc[:,'freq']),
            #b_fft=np.fft.fft(full_b),
            b_fft_mean=np.mean(np.abs(np.fft.fft(full_b)),axis=1),
            #b_peak2peak=full_b.max(axis=1) - full_b.min(axis=1),
            log_peak2peak = np.log(full_b.max(axis=1) - full_b.min(axis=1)),
            #max_dbdt=np.max(dbdt, axis=1),
            #min_dbdt=np.min(dbdt, axis=1),
            log_mean_abs_dbdt=np.log(np.mean(np.abs(dbdt), axis=1)),
            #crest_fac=crest_factor(full_b),
            form_fac=form_factor(full_b),
            # median_dbdt=np.median(dbdt, axis=1)
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
        cols= x_cols
        for kfold_lbl, test_fold_df in mat_df_proc.groupby("kfold"):
            train_fold_df = (
                mat_df_proc.query("kfold != @kfold_lbl")
                .reset_index(drop=True)
                .drop(columns="kfold")
            )
            assert len(train_fold_df) > 0, "empty dataframe error"
            #y = np.log(train_fold_df.pop("ploss"))
            y = train_fold_df.pop("ploss").to_numpy()
            X = train_fold_df.loc[:, x_cols].to_numpy()

            gbm = xgboost.XGBRegressor(max_depth=12, gamma = 0.05822,learning_rate=0.1, n_estimators=850, subsample=1, colsample_bytree=1, objective='reg:squarederror')
            gbm.fit(X, y)
            #pred = np.exp(gbm.predict(test_fold_df.loc[:, x_cols]))
            pred= gbm.predict(test_fold_df.loc[:, x_cols])
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
        plt.bar(range(len(feature_imp_sum)), feature_imp_sum)
        plt.xticks(range(len(feature_imp_sum)), cols, rotation='vertical')
        plt.show()
        print(feature_imp_sum)
    else:
        print(pd.DataFrame(exp_log).T.loc[:, ["avg-abs-rel-err"]])
    #print(pd.DataFrame(exp_log).T)
    #print(np.mean(pd.DataFrame(exp_log).T.loc[:,"avg-abs-rel-err"].to_numpy()))
    #plt.bar(range(len(feature_imp_sum)), feature_imp_sum)
    #plt.xticks(range(len(feature_imp_sum)), cols, rotation='horizontal')
    #plt.show()



if __name__ == "__main__":
    main()
