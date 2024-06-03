import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Latex
from utils import wandb_utils
from exp import data_backup

saved_data = data_backup.backup_experiment_data(
    experiment_tags = [
        "e32_icml_erm_newdatasets_newmodels",
        "e33_icml_constrained_newdatasets_newmodels", # has train and val
        "e29_icml_erm_newdatasets",
        "e30_icml_constrained_newdatasets_train",
        "e31_icml_constrained_newdatasets_val",
        "e29_icml_constrained_newmodels",
        "erm_rebuttal_newmodels",
        'e34_icml_resilience_newdatasets_newmodels',
        "e35_icml_const_and_resilience_gaps"
    ],
    filename_prefix="data_backups/2024_05_14__rebuttal_all_new_erm_and_constrained"
)
