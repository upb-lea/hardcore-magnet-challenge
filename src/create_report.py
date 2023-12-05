import pandas as pd
import argparse
import matplotlib.pyplot as plt
from utils.data import (
    PROC_SOURCE,
    PRED_SINK,
    EXP_CSV_COLS,
    EXP_CSV_PATH,
    TRIALS_CSV_COLS,
    TRIALS_CSV_PATH,
    load_new_materials,
)
from utils.visualization import visualize_rel_error_report

parser = argparse.ArgumentParser(description="Create Report PDF from experiment ID")
parser.add_argument("experiment_id", help="The experiment uid")

args = parser.parse_args()

# load up meta info
exp_tab = pd.read_csv(EXP_CSV_PATH, dtype=EXP_CSV_COLS)
trials_tab = pd.read_csv(
    TRIALS_CSV_PATH,
    dtype=TRIALS_CSV_COLS,
    parse_dates=["start_date", "end_date"],
).query(f"experiment_uid == '{args.experiment_id}'")
# only one row should be returned
meta_info_df = trials_tab.merge(exp_tab, on="experiment_uid")

mean_fold_score_df = (
    meta_info_df.groupby(["material", "seed"])["95-perc_rel_err"]
    .agg("mean")
    .reset_index()
)
best_seed_per_material_df = mean_fold_score_df.loc[
    mean_fold_score_df.groupby("material")["95-perc_rel_err"].idxmin(), :
]
print("Best seed per material:")
print(best_seed_per_material_df)

# the same for each experiment
is_p_predictor = meta_info_df.predicts_p_directly.iloc[0]
was_debug = meta_info_df.debug.iloc[0]
on_old_mats = "3C90" in meta_info_df.material.unique()
# load up predictions
preds = pd.read_csv(
    PRED_SINK
    / f"CNN_{'P' if is_p_predictor else 'H'}_preds_{args.experiment_id}{'_debug' if was_debug else ''}.csv.zip",
    dtype={"material": str},
).rename(columns={"pred": "ploss"})

targeted_materials = meta_info_df.material.unique().tolist()
if on_old_mats:
    ds = pd.read_pickle(
        PROC_SOURCE / f"ten_materials.pkl.gz"
    )  # .query("material in @targeted_material")
else:
    ds = load_new_materials(training=True, filter_materials=targeted_materials)

if was_debug:
    ds = pd.concat(
        [
            df.iloc[: preds.shape[0] // len(targeted_materials), :]
            for m_lbl, df in ds.groupby("material")
        ],
        ignore_index=True,
    )

fig = visualize_rel_error_report(
    preds, gtruth=ds, title="Magnet Challenge 2023 - Team Paderborn"
)
plt.show()

plt.savefig(f"{PRED_SINK}/{args.experiment_id}.pdf")