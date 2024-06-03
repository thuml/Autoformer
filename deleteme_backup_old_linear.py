import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Latex
from utils import wandb_utils
from exp import data_backup



saved_data = data_backup.backup_experiment_data(experiment_tags=["e16_finaljan_allerm_replicated","e21_icml_static_linear_no_resilience"],
            filename_prefix="data_backups/2024_06_03__camera_ready_constant_linear_oldruns")