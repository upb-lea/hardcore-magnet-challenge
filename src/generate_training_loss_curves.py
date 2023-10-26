import pandas as pd
from matplotlib import pyplot as plt
import os

# Generate PDF files to show the training and validation losses from .csv data taken when training the model.
# As a directory, the model directory needs to be given which include the .csv files from the training.


#directory = '/home/nikolasf/Dokumente/01_git/30_Python/MC_UPB/data/models/kfold_4_subsamples_4_graphs'
directory = '/home/nikolasf/Dokumente/01_git/30_Python/MC_UPB/data/models/kfold_4_subsamples_4_batch_16'



for file_name in os.listdir(directory):
    if '.csv' in file_name:
        df_read = pd.read_csv(os.path.join(directory, file_name))

        plt.plot(df_read.index, df_read['training loss'], label='training loss', color='blue')
        plt.plot(df_read.index, df_read['validation loss'], label='validation loss', color='red', marker='o')
        plt.xlabel('epochs')
        plt.ylabel('training / validation loss')
        plt.grid()
        plt.title(file_name)
        plt.legend()
        plt.ylim([0, 0.0002])
        save_path = directory + '/' + file_name.replace('.csv', '.pdf')
        print(f"{save_path = }")
        plt.savefig(save_path)
        plt.close()

