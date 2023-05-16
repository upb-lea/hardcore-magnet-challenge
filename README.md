# magnet-challenge-2023


Folder structure:
```
├── data
│   ├── input   
│   │   ├── processed  # all processings of the raw data set
│   │   └── raw   # initial data set
│   └── output  # model dumps, estimation arrays, model architectures, etc.
├── docs
│   ├── EDA_24th_April_2023.pdf  # all documents, papers, etc.
│   └── webinar_kickoff.pdf
├── notebooks
│   ├── 1.0-wk-process_raw.ipynb  # number-author_initials-topic.ipynb
│   └── 1.1-wk-eda.ipynb
├── src
│   └── wk_runner_script.py   # python scripts
└── README.md

```

## How to get rollin'

* Get the full data set here: [Princeton Website](https://mag-net.princeton.edu/)
* Extract the material folders into data/input/raw such that you get a folder structure like

```
data/
├── input
│   ├── processed
│   └── raw
│       ├── 3C90
│       │   ├── B_waveform[T].csv
│       │   ├── Frequency[Hz].csv
│       │   ├── H_waveform[Am-1].csv
│       │   ├── Temperature[C].csv
│       │   └── Volumetric_losses[Wm-3].csv
│       ├── 3C94
│       │   ├── B_waveform[T].csv
│       │   ├── Frequency[Hz].csv
│       │   ├── H_waveform[Am-1].csv
│       │   ├── Temperature[C].csv
│       │   └── Volumetric_losses[Wm-3].csv
..      ..      ..

```

* Execute `notebooks/1.0-wk-process_raw.ipynb`, the processed pickle file will land in `data/input/processed`
* Alternatively, ask Wilhelm for the processed file

## Tasks during the project
Work packages are tracked at [LEA cloud](https://cloud.lea-com.upb.de/index.php/apps/deck/#/board/47)

