<div align="center">
<h1>MagNet Challenge 2023 Paderborn University Solution </h1>
<h3>HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual, dilated convolutional neural networks in ferrite cores
</h3>

[Wilhelm Kirchgässner](https://github.com/wkirgsn), [Nikolas Förster](https://github.com/gituser789), [Till Piepenbrock](https://github.com/tillpiepenbrock), [Oliver Schweins](https://github.com/OliverSchw), [Oliver Wallscheid](https://github.com/wallscheid)

[Paderborn University](https://www.uni-paderborn.de/en/), [Dept. of Power Electronics and Electrical Drives](https://ei.uni-paderborn.de/en/lea)

Paper: [arXiv 2401.10166](https://arxiv.org/abs/2401.11488)

</div>

## Updates
* **` Jan. 27th, 2024`:** Publish the submission on GitHub

## Abstract
The [MagNet Challenge 2023](https://github.com/minjiechen/magnetchallenge) calls upon competitors to develop data-driven models for the material-specific, waveform-agnostic estimation of steady-state power losses in toroidal ferrite cores. The following HARDCORE (H-field and power loss estimation for Arbitrary waveforms with Residual, Dilated convolutional neural networks in ferrite COREs) approach shows that a residual convolutional neural network with physics-informed extensions can serve this task efficiently when trained on observational data beforehand. One key solution element is an intermediate model layer which first reconstructs the bh curve and then estimates the power losses based on the curve's area rendering the proposed topology physically interpretable. In addition, emphasis was placed on expert-based feature engineering and information-rich inputs in order to enable a lean model architecture. A model is trained from scratch for each material, while the topology remains the same. A Pareto-style trade-off between model size and estimation accuracy is demonstrated, which yields an optimum at as low as 1755 parameters and down to below 8\,\% for the 95-th percentile of the relative error for the worst-case material with sufficient samples.


## Overview
Folder structure:

```
.
├── Model
│   ├── data
│   │   ├── b_max_dict.json
│   │   ├── h_max_dict.json
│   │   ├── input
│   │   │   └── test
│   │   │       ├── place_Testing_here.txt
│   │   │       └── Testing
│   │   │           ├── Material A
│   │   │           │   ├── B_Field.csv
│   │   │           │   ├── Frequency.csv
│   │   │           │   └── Temperature.csv
│   │   │           ├── Material B
│   │   │           │   ├── B_Field.csv
│   │   │           │   ├── Frequency.csv
│   │   │           │   └── Temperature.csv
│   │   │           ├── Material C
│   │   │           │   ├── B_Field.csv
│   │   │           │   ├── Frequency.csv
│   │   │           │   └── Temperature.csv
│   │   │           ├── Material D
│   │   │           │   ├── B_Field.csv
│   │   │           │   ├── Frequency.csv
│   │   │           │   └── Temperature.csv
│   │   │           └── Material E
│   │   │               ├── B_Field.csv
│   │   │               ├── Frequency.csv
│   │   │               └── Temperature.csv
│   │   ├── models
│   │   │   ├── cnn_A_experiment_c9cfe_model_d893c778_seed_0_fold_0.pt
│   │   │   ├── cnn_B_experiment_c9cfe_model_b6a920cc_seed_0_fold_0.pt
│   │   │   ├── cnn_C_experiment_c9cfe_model_c1ced7b6_seed_0_fold_0.pt
│   │   │   ├── cnn_D_experiment_c9cfe_model_11672810_seed_0_fold_0.pt
│   │   │   └── cnn_E_experiment_c9cfe_model_5ae50f9e_seed_0_fold_0.pt
│   │   └── output
│   │       ├── experiments_meta.csv
│   │       └── trials_meta.csv
│   └── src
│       ├── Model_Inference.py
│       ├── run_cnn_inference.py
│       ├── run_cnn_training.py
│       └── utils
│           ├── data.py
│           ├── experiments.py
│           ├── metrics.py
│           ├── topology.py
│           └── visualization.py
├── PaderbornUniversity_Report.pdf
├── README.md
├── requirements.txt
└── Result
    ├── Volumetric_Loss_Material A.csv
    ├── Volumetric_Loss_Material B.csv
    ├── Volumetric_Loss_Material C.csv
    ├── Volumetric_Loss_Material D.csv
    └── Volumetric_Loss_Material E.csv


```

Note that the folder `Testing/`, which contains all the final test data of the new 5 materials, has to be moved into the designated folder, as shown above.


## How to set up the Python environment

We highly recommend using a Python virtual environment, such as venv or Anaconda.
We recommend Python 3.10 for this project.
When the virtual environment shell is active and the current working directory is the project root, install the dependencies with 

```
>>> pip install -r requirements.txt
```

## How to execute the model inference

The main inference script is `Model/src/Model_Inference.py` which was provided as template by the competition hosts.
Execute it while the current working directory is `Model/src/' like below

```py
>>> cd Model/src/
>>> python Model_Inference.py

```

The current working directory is important for all relative paths to work flawlessly.
Note that only one material is estimated per call. In order to estimate another material, edit the file `Model_Inference.py` as intended by the template.
All estimates are stored under `Result/`. For reference, the inference result as obtained on our computing machines is provided in this submission as well.

The inference script will load up the corresponding model under `Model/data/models`, which are all saved through PyTorch jit functionality.
Next to the requested estimation csv under `Result/`, there is also a slightly different format for the loss estimates stored under `Model/data/output/` together with a corresponding H-field estimation. Feel free to analyze, skim through, or just double-check the H-field and p loss estimates there.
