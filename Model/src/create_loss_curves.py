import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
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

parser = argparse.ArgumentParser(description="Create loss curves from experiment ID")
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
    meta_info_df.groupby(["material", "seed", 'tag'])["95-perc_rel_err"]
    .agg("mean")
    .reset_index()
)
best_seed_per_material_df = mean_fold_score_df.loc[
    mean_fold_score_df.groupby("material")["95-perc_rel_err"].idxmin(), :
]
print("Best seed per material:")
print(best_seed_per_material_df)
# load learning curve csv files:
learning_curves = pd.read_csv(f"{PRED_SINK}/learning_curves_cnn_{args.experiment_id}.csv.zip")

number_of_folds = meta_info_df.loc[0, 'n_folds']

for material in best_seed_per_material_df["material"]:

    seed_list = np.arange(0, 20)

    for seed in seed_list:

        try:
            # figure out best seed
            best_seed = best_seed_per_material_df.loc[best_seed_per_material_df["material"] == material]["seed"].to_numpy(dtype=int)[0]
            materials_learning_curves = learning_curves.loc[(learning_curves["material"] == material) & (learning_curves["seed"] == seed)]

            # print(f"{materials_learning_curves = }")

            if materials_learning_curves.empty:
                break

            number_epochs = len(materials_learning_curves.index)
            count_epochs = range(0, number_epochs)
            experiment_tag = best_seed_per_material_df.loc[best_seed_per_material_df["material"] == material]["tag"].to_string(index=False)

            fig, axes = plt.subplots(
                nrows=np.ceil(number_of_folds / 2).astype(int),
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(8.26, 8.26),
            )

            for fold, ax in zip(range(0, number_of_folds + 1), axes.flatten()):
                ax.semilogy(count_epochs, materials_learning_curves[f"loss_trends_train_h_fold_{fold}"], label=f"train_h")
                ax.semilogy(count_epochs, materials_learning_curves[f"loss_trends_train_p_fold_{fold}"], label=f"train_p")
                ax.semilogy(count_epochs, materials_learning_curves[f"loss_trends_val_h_fold_{fold}"], label=f"val_h", marker="o", markersize=1)
                ax.semilogy(count_epochs, materials_learning_curves[f"loss_trends_val_p_fold_{fold}"], label=f"val_p", marker="o", markersize=1)
                ax.set_xlabel("epochs")
                ax.set_ylabel("training or validation loss")
                ax.set_title(f"Fold {fold}")
                ax.legend()
                ax.grid(which='both', axis='both', visible=True)



            fig.suptitle(
                f"Eperiment {args.experiment_id}, Material {material}, Seed {seed} \n"
                f"{experiment_tag}",
                fontweight="bold",
            )
            plt.tight_layout()
            # plt.show()
            if seed == best_seed:
                plt.savefig(PRED_SINK / f"LearnCurve_Exp_{args.experiment_id}_Material_{material}_Seed_{seed}_best_seed.pdf")
            else:
                plt.savefig(PRED_SINK / f"LearnCurve_Exp_{args.experiment_id}_Material_{material}_Seed_{seed}.pdf")
        except:
            pass
