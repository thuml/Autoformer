"""
Script to run sweeps 2.0. Before, we used YAMLs. Now we use the wandb API on python.
"""
import yaml
import os
import wandb
from functools import reduce

###### SCRIPT PARAMETERS

WANDB_PROJECT="Autoformer"
#WANDB_PROJECT="Autoformer-javierdev"
NAMESPACE="alelab"
YAML_DEBUG_LOCATION="../generated_sweeps/"
if not os.path.exists(YAML_DEBUG_LOCATION):
    os.makedirs(YAML_DEBUG_LOCATION)

SWEEP_NAME_PREFIX="ICML_CONSTR_RES_GAPS"
EXPERIMENT_DESCRIPTION='Constrained And Resilient Gaps'
#EXPERIMENT_TAG="e34_icml_resilience_newdatasets_newmodels"
EXPERIMENT_TAG="e35_icml_const_and_resilience_gaps"

# Constraint parameters
# Constant
#CONSTRAINT_TYPE='resilience'
CONSTRAINT_TYPE='constant'
#CONSTRAINT_TYPE='monotonic'
#CONSTRAINT_TYPE='static_linear'
DUAL_LR=0.01
DUAL_INIT=1.0

# CONSTRAINT_TYPE='erm'
# DUAL_LR=0.0
# DUAL_INIT=0.0

# Required if Resilience
# RESILIENT_LR=0.1
RESILIENT_LR=0.0

#PROD PARAMETERS
# OLD MODELS
#MODELS = ["Pyraformer"]#["Autoformer","Reformer","Informer","Transformer"]
# NEW MODELS
# MODELS = [
  #"Pyraformer",
  # "Nonstationary_Transformer",
  # "iTransformer",
  #"FiLM"
# ]

# ALLMODELS
MODELS = [
  # "Autoformer",
  # "Reformer",
  # "Informer",
  # "Transformer",
  "Pyraformer",
  "Nonstationary_Transformer",
  "iTransformer",
  #  "FiLM"
]


#MODELS = ["PatchTST"] #Must run by itself
# MODELS = ["Koopa"]
if len(MODELS)>1 and "PatchTST" in MODELS:
    raise ValueError("PatchtTST Must be run separately because of its unique parameters")
DATASETS=[
  "weather.csv",
  # "electricity.csv",
  # "exchange_rate.csv",
  # "traffic.csv",
  # "ETTh1.csv", 
  # "ETTh2.csv",
  # "ETTm1.csv",
  # "ETTm2.csv",
  # "national_illness.csv"
]
# The script will use one of these based on the dataset. You can change these lists to only run for some pred lengths.
STANDARD_PRED_LENGTHS = [
96,
192,
336,
720
]

ILLNESS_PRED_LENGTHS = [
  24,
 36,
 48,
 60] # For illness dataset
  
NUM_SEEDS=1
# SEED=0 # IF 0 THEN RANDOM SEED
SEED=2021 # The seed used by the literature
# SEED=6163 # Seed 2.
# SEED=3589 # Seed 3.

#END PARAMETERS
###################
###################
print(f"Creating sweeps in project {WANDB_PROJECT}, namespace {NAMESPACE}")

# Only used by ERM, will be overwritten in constrained
if len(MODELS)>1:
    MODEL_DICT = {"values": MODELS}
else:
    MODEL_DICT = {"value": MODELS[0]}

DATASET_DEPENDENT={
  "weather.csv":{
    'root_path': {'value': './dataset/weather/'},
    'data_path': {'value': 'weather.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 21},
    'dec_in': {'value': 21},
    'c_out': {'value': 21},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "electricity.csv":{
    'root_path': {'value': './dataset/electricity/'},
    'data_path': {'value': 'electricity.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 321},
    'dec_in': {'value': 321},
    'c_out': {'value': 321},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "exchange_rate.csv":{
    'root_path': {'value': './dataset/exchange_rate/'},
    'data_path': {'value': 'exchange_rate.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 8},
    'dec_in': {'value': 8},
    'c_out': {'value': 8},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "traffic.csv":{
    'root_path': {'value': './dataset/traffic/'},
    'data_path': {'value': 'traffic.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 862},
    'dec_in': {'value': 862},
    'c_out': {'value': 862},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "national_illness.csv":{
    'root_path': {'value': './dataset/illness/'},
    'data_path': {'value': 'national_illness.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 7},
    'dec_in': {'value': 7},
    'c_out': {'value': 7},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "ETTh1.csv":{ 
    'root_path': {'value': './dataset/ETT-small/'},
    'data_path': {'value': 'ETTh1.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 7},
    'dec_in': {'value': 7},
    'c_out': {'value': 7},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "ETTh2.csv":{ 
    'root_path': {'value': './dataset/ETT-small/'},
    'data_path': {'value': 'ETTh2.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 7},
    'dec_in': {'value': 7},
    'c_out': {'value': 7},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
  },
  "ETTm1.csv":{ 
    'root_path': {'value': './dataset/ETT-small/'},
    'data_path': {'value': 'ETTm1.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 7},
    'dec_in': {'value': 7},
    'c_out': {'value': 7},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
    'freq': {'value':'t'},
  },
  "ETTm2.csv":{ 
    'root_path': {'value': './dataset/ETT-small/'},
    'data_path': {'value': 'ETTm2.csv'},
    'model_id': {'value': 'placeholder'},
    'enc_in': {'value': 7},
    'dec_in': {'value': 7},
    'c_out': {'value': 7},
    'features': {'value': 'M'},
    'data': {'value': 'custom'},
    'freq': {'value':'t'},
  },
}

# Train constraints
# CONSTRAINT_DATA={'ETTh1.csv': {'Autoformer': {96: [0.246, 0.262, 0.29],
#    192: [0.306, 0.322, 0.332],
#    336: [0.346, 0.353, 0.357],
#    720: [0.454, 0.472, 0.505]},
#   'FiLM': {96: [0.332, 0.401, 0.457],
#    192: [0.404, 0.503, 0.588],
#    336: [0.487, 0.628, 0.696],
#    720: [0.669, 0.768, 0.877]},
#   'Informer': {96: [0.257, 0.263, 0.27],
#    192: [0.305, 0.351, 0.392],
#    336: [0.412, 0.426, 0.433],
#    720: [0.436, 0.446, 0.456]},
#   'Nonstationary_Transformer': {96: [0.128, 0.133, 0.135],
#    192: [0.166, 0.168, 0.17],
#    336: [0.189, 0.191, 0.194],
#    720: [0.226, 0.233, 0.236]},
#   'Pyraformer': {96: [0.249, 0.261, 0.27],
#    192: [0.28, 0.284, 0.288],
#    336: [0.297, 0.301, 0.306],
#    720: [0.33, 0.336, 0.34]},
#   'Reformer': {96: [0.283, 0.316, 0.34],
#    192: [0.315, 0.325, 0.334],
#    336: [0.346, 0.355, 0.369],
#    720: [0.388, 0.4, 0.411]},
#   'Transformer': {96: [0.149, 0.152, 0.155],
#    192: [0.171, 0.172, 0.174],
#    336: [0.187, 0.188, 0.19],
#    720: [0.207, 0.208, 0.21]},
#   'iTransformer': {96: [0.249, 0.298, 0.337],
#    192: [0.32, 0.379, 0.434],
#    336: [0.398, 0.459, 0.505],
#    720: [0.51, 0.572, 0.634]}},
#  'ETTh2.csv': {'Autoformer': {96: [0.167, 0.184, 0.206],
#    192: [0.252, 0.274, 0.316],
#    336: [0.303, 0.323, 0.359],
#    720: [0.502, 0.562, 0.634]},
#   'FiLM': {96: [0.247, 0.337, 0.387],
#    192: [0.34, 0.421, 0.478],
#    336: [0.413, 0.509, 0.598],
#    720: [0.544, 0.713, 0.878]},
#   'Informer': {96: [0.133, 0.137, 0.143],
#    192: [0.151, 0.154, 0.158],
#    336: [0.164, 0.168, 0.174],
#    720: [0.183, 0.188, 0.2]},
#   'Nonstationary_Transformer': {96: [0.086, 0.099, 0.114],
#    192: [0.112, 0.145, 0.169],
#    336: [0.147, 0.194, 0.239],
#    720: [0.227, 0.329, 0.401]},
#   'Pyraformer': {96: [0.135, 0.148, 0.157],
#    192: [0.158, 0.168, 0.176],
#    336: [0.183, 0.191, 0.201],
#    720: [0.222, 0.235, 0.253]},
#   'Reformer': {96: [0.162, 0.18, 0.2],
#    192: [0.186, 0.202, 0.209],
#    336: [0.197, 0.204, 0.218],
#    720: [0.218, 0.225, 0.228]},
#   'Transformer': {96: [0.088, 0.089, 0.09],
#    192: [0.105, 0.106, 0.108],
#    336: [0.12, 0.121, 0.123],
#    720: [0.142, 0.144, 0.145]},
#   'iTransformer': {96: [0.154, 0.199, 0.24],
#    192: [0.22, 0.297, 0.338],
#    336: [0.306, 0.38, 0.459],
#    720: [0.419, 0.575, 0.729]}},
#  'ETTm1.csv': {'Autoformer': {96: [0.119, 0.123, 0.145],
#    192: [0.185, 0.216, 0.235],
#    336: [0.243, 0.271, 0.299],
#    720: [0.313, 0.342, 0.378]},
#   'FiLM': {96: [0.303, 0.335, 0.343],
#    192: [0.333, 0.348, 0.406],
#    336: [0.346, 0.412, 0.499],
#    720: [0.416, 0.536, 0.628]},
#   'Informer': {96: [0.116, 0.124, 0.132],
#    192: [0.127, 0.132, 0.14],
#    336: [0.138, 0.143, 0.15],
#    720: [0.163, 0.167, 0.176]},
#   'Nonstationary_Transformer': {96: [0.054, 0.058, 0.059],
#    192: [0.07, 0.073, 0.08],
#    336: [0.083, 0.092, 0.098],
#    720: [0.115, 0.123, 0.132]},
#   'Pyraformer': {96: [0.131, 0.143, 0.155],
#    192: [0.165, 0.175, 0.184],
#    336: [0.192, 0.204, 0.211],
#    720: [0.23, 0.237, 0.243]},
#   'Reformer': {96: [0.166, 0.175, 0.179],
#    192: [0.177, 0.181, 0.192],
#    336: [0.187, 0.19, 0.194],
#    720: [0.206, 0.209, 0.219]},
#   'Transformer': {96: [0.049, 0.05, 0.053],
#    192: [0.059, 0.06, 0.062],
#    336: [0.067, 0.068, 0.07],
#    720: [0.082, 0.084, 0.085]},
#   'iTransformer': {96: [0.164, 0.191, 0.201],
#    192: [0.218, 0.226, 0.278],
#    336: [0.243, 0.292, 0.332],
#    720: [0.325, 0.379, 0.423]}},
#  'ETTm2.csv': {'Autoformer': {96: [0.109, 0.129, 0.152],
#    192: [0.165, 0.191, 0.224],
#    336: [0.204, 0.26, 0.309],
#    720: [0.291, 0.355, 0.393]},
#   'FiLM': {96: [0.133, 0.184, 0.207],
#    192: [0.185, 0.223, 0.309],
#    336: [0.214, 0.327, 0.407],
#    720: [0.332, 0.448, 0.513]},
#   'Informer': {96: [0.065, 0.07, 0.075],
#    192: [0.07, 0.074, 0.077],
#    336: [0.074, 0.077, 0.081],
#    720: [0.087, 0.089, 0.094]},
#   'Nonstationary_Transformer': {96: [0.038, 0.043, 0.049],
#    192: [0.052, 0.061, 0.073],
#    336: [0.066, 0.086, 0.099],
#    720: [0.102, 0.124, 0.192]},
#   'Pyraformer': {96: [0.074, 0.084, 0.092],
#    192: [0.09, 0.096, 0.111],
#    336: [0.102, 0.11, 0.118],
#    720: [0.123, 0.133, 0.136]},
#   'Reformer': {96: [0.084, 0.096, 0.104],
#    192: [0.093, 0.096, 0.101],
#    336: [0.092, 0.094, 0.098],
#    720: [0.107, 0.111, 0.123]},
#   'Transformer': {96: [0.038, 0.038, 0.039],
#    192: [0.043, 0.043, 0.044],
#    336: [0.048, 0.048, 0.048],
#    720: [0.053, 0.054, 0.054]},
#   'iTransformer': {96: [0.083, 0.108, 0.128],
#    192: [0.123, 0.157, 0.191],
#    336: [0.17, 0.223, 0.261],
#    720: [0.245, 0.313, 0.408]}},
#  'electricity.csv': {'Autoformer': {96: [0.111, 0.115, 0.118],
#    192: [0.139, 0.144, 0.153],
#    336: [0.151, 0.159, 0.163],
#    720: [0.176, 0.183, 0.19]},
#   'FiLM': {96: [0.197, 0.242, 0.26],
#    192: [0.244, 0.259, 0.269],
#    336: [0.259, 0.275, 0.314],
#    720: [0.295, 0.369, 0.451]},
#   'Informer': {96: [0.146, 0.149, 0.156],
#    192: [0.164, 0.168, 0.17],
#    336: [0.186, 0.192, 0.197],
#    720: [0.206, 0.208, 0.212]},
#   'Nonstationary_Transformer': {96: [0.059, 0.06, 0.062],
#    192: [0.071, 0.072, 0.074],
#    336: [0.078, 0.079, 0.081],
#    720: [0.088, 0.089, 0.091]},
#   'Pyraformer': {96: [0.14, 0.142, 0.143],
#    192: [0.148, 0.149, 0.152],
#    336: [0.153, 0.154, 0.156],
#    720: [0.158, 0.16, 0.162]},
#   'Reformer': {96: [0.151, 0.152, 0.153],
#    192: [0.16, 0.16, 0.161],
#    336: [0.165, 0.166, 0.167],
#    720: [0.176, 0.176, 0.178]},
#   'Transformer': {96: [0.107, 0.108, 0.109],
#    192: [0.114, 0.116, 0.121],
#    336: [0.127, 0.129, 0.134],
#    720: [0.138, 0.139, 0.142]},
#   'iTransformer': {96: [0.123, 0.142, 0.144],
#    192: [0.145, 0.149, 0.156],
#    336: [0.151, 0.165, 0.191],
#    720: [0.187, 0.208, 0.231]}},
#  'exchange_rate.csv': {'Autoformer': {96: [0.107, 0.15, 0.214],
#    192: [0.232, 0.266, 0.336],
#    336: [0.25, 0.33, 0.46],
#    720: [0.411, 0.85, 1.173]},
#   'FiLM': {96: [0.184, 0.254, 0.308],
#    192: [0.231, 0.36, 0.479],
#    336: [0.478, 0.701, 0.893],
#    720: [0.651, 1.116, 1.405]},
#   'Informer': {96: [0.037, 0.038, 0.041],
#    192: [0.04, 0.042, 0.045],
#    336: [0.046, 0.048, 0.053],
#    720: [0.051, 0.055, 0.062]},
#   'Nonstationary_Transformer': {96: [0.017, 0.021, 0.03],
#    192: [0.029, 0.042, 0.06],
#    336: [0.178, 0.294, 0.412],
#    720: [0.08, 0.139, 0.22]},
#   'Pyraformer': {96: [0.039, 0.043, 0.043],
#    192: [0.188, 0.206, 0.222],
#    336: [0.219, 0.233, 0.256],
#    720: [0.06, 0.063, 0.065]},
#   'Reformer': {96: [0.079, 0.108, 0.139],
#    192: [0.134, 0.189, 0.269],
#    336: [0.209, 0.349, 0.46],
#    720: [0.371, 0.608, 0.682]},
#   'Transformer': {96: [0.025, 0.028, 0.029],
#    192: [0.029, 0.031, 0.035],
#    336: [0.037, 0.039, 0.042],
#    720: [0.048, 0.049, 0.051]},
#   'iTransformer': {96: [0.062, 0.115, 0.176],
#    192: [0.12, 0.241, 0.356],
#    336: [0.27, 0.476, 0.676],
#    720: [0.336, 0.827, 1.192]}},
#  'national_illness.csv': {'Autoformer': {24: [0.359, 0.413, 0.434],
#    36: [0.392, 0.44, 0.461],
#    48: [0.498, 0.52, 0.533],
#    60: [0.437, 0.453, 0.467]},
#   'FiLM': {24: [0.732, 0.76, 0.787],
#    36: [0.88, 0.927, 0.941],
#    48: [0.827, 0.856, 0.9],
#    60: [0.76, 0.796, 0.818]},
#   'Informer': {24: [0.296, 0.316, 0.331],
#    36: [0.308, 0.329, 0.348],
#    48: [0.325, 0.355, 0.382],
#    60: [0.35, 0.369, 0.4]},
#   'Nonstationary_Transformer': {24: [0.36, 0.38, 0.403],
#    36: [0.36, 0.396, 0.419],
#    48: [0.445, 0.465, 0.474],
#    60: [0.412, 0.457, 0.473]},
#   'Pyraformer': {24: [0.296, 0.301, 0.31],
#    36: [0.321, 0.327, 0.336],
#    48: [0.342, 0.352, 0.374],
#    60: [0.372, 0.385, 0.41]},
#   'Reformer': {24: [0.499, 0.506, 0.521],
#    36: [0.515, 0.519, 0.523],
#    48: [0.513, 0.543, 0.554],
#    60: [0.518, 0.553, 0.559]},
#   'Transformer': {24: [0.278, 0.308, 0.361],
#    36: [0.342, 0.367, 0.4],
#    48: [0.403, 0.42, 0.429],
#    60: [0.421, 0.474, 0.481]},
#   'iTransformer': {24: [0.314, 0.383, 0.395],
#    36: [0.4, 0.413, 0.42],
#    48: [0.419, 0.442, 0.453],
#    60: [0.442, 0.462, 0.479]}},
#  'traffic.csv': {'Autoformer': {96: [0.205, 0.206, 0.208],
#    192: [0.222, 0.227, 0.233],
#    336: [0.23, 0.234, 0.238],
#    720: [0.254, 0.259, 0.267]},
#   'FiLM': {96: [0.246, 0.252, 0.253],
#    192: [0.242, 0.245, 0.246],
#    336: [0.247, 0.265, 0.269],
#    720: [0.295, 0.323, 0.342]},
#   'Informer': {96: [0.238, 0.245, 0.253],
#    192: [0.259, 0.263, 0.269],
#    336: [0.294, 0.299, 0.306],
#    720: [0.424, 0.435, 0.445]},
#   'Nonstationary_Transformer': {96: [0.257, 0.259, 0.261],
#    192: [0.262, 0.265, 0.268],
#    336: [0.268, 0.275, 0.28],
#    720: [0.287, 0.295, 0.306]},
#   'Pyraformer': {96: [0.256, 0.258, 0.265],
#    192: [0.269, 0.273, 0.277],
#    336: [0.273, 0.275, 0.279],
#    720: [0.279, 0.281, 0.286]},
#   'Reformer': {96: [0.249, 0.25, 0.253],
#    192: [0.256, 0.257, 0.259],
#    336: [0.264, 0.266, 0.267],
#    720: [0.278, 0.28, 0.285]},
#   'Transformer': {96: [0.206, 0.207, 0.209],
#    192: [0.211, 0.212, 0.215],
#    336: [0.221, 0.222, 0.224],
#    720: [0.23, 0.232, 0.236]},
#   'iTransformer': {96: [0.22, 0.23, 0.231],
#    192: [0.23, 0.233, 0.24],
#    336: [0.234, 0.247, 0.259],
#    720: [0.253, 0.269, 0.279]}},
#  'weather.csv': {'Autoformer': {96: [0.303, 0.308, 0.358],
#    192: [0.383, 0.464, 0.696],
#    336: [0.484, 0.541, 0.602],
#    720: [0.67, 0.732, 0.759]},
#   'FiLM': {96: [0.507, 0.557, 0.564],
#    192: [0.556, 0.567, 0.618],
#    336: [0.566, 0.697, 0.709],
#    720: [0.699, 0.754, 0.792]},
#   'Informer': {96: [0.213, 0.217, 0.222],
#    192: [0.222, 0.227, 0.234],
#    336: [0.232, 0.237, 0.245],
#    720: [0.245, 0.249, 0.26]},
#   'Nonstationary_Transformer': {96: [0.274, 0.288, 0.293],
#    192: [0.272, 0.274, 0.284],
#    336: [0.293, 0.308, 0.31],
#    720: [0.356, 0.365, 0.375]},
#   'Pyraformer': {96: [0.324, 0.348, 0.362],
#    192: [0.279, 0.283, 0.295],
#    336: [0.312, 0.325, 0.329],
#    720: [0.36, 0.368, 0.371]},
#   'Reformer': {96: [0.29, 0.303, 0.32],
#    192: [0.306, 0.315, 0.327],
#    336: [0.308, 0.312, 0.315],
#    720: [0.281, 0.282, 0.283]},
#   'Transformer': {96: [0.164, 0.167, 0.173],
#    192: [0.178, 0.182, 0.189],
#    336: [0.194, 0.195, 0.198],
#    720: [0.213, 0.215, 0.216]},
#   'iTransformer': {96: [0.348, 0.408, 0.439],
#    192: [0.418, 0.47, 0.501],
#    336: [0.475, 0.54, 0.599],
#    720: [0.578, 0.653, 0.682]}}}

# Val Constraints
CONSTRAINT_DATA={'ETTh1.csv': {'Autoformer': {96: [0.507, 0.521, 0.533],
   192: [0.431, 0.46, 0.496],
   336: [0.539, 0.569, 0.592],
   720: [0.554, 0.591, 0.719]},
  'FiLM': {96: [0.327, 0.369, 0.408],
   192: [0.382, 0.437, 0.456],
   336: [0.448, 0.468, 0.497],
   720: [0.425, 0.468, 0.498]},
  'Informer': {96: [0.713, 1.015, 1.115],
   192: [0.907, 1.007, 1.073],
   336: [0.822, 0.897, 1.055],
   720: [0.853, 0.966, 1.199]},
  'Nonstationary_Transformer': {96: [0.422, 0.451, 0.459],
   192: [0.481, 0.531, 0.596],
   336: [0.577, 0.639, 0.743],
   720: [0.579, 0.648, 0.72]},
  'Pyraformer': {96: [0.363, 0.43, 0.5],
   192: [0.458, 0.544, 0.618],
   336: [0.547, 0.692, 0.895],
   720: [0.881, 1.053, 1.187]},
  'Reformer': {96: [0.677, 0.78, 0.828],
   192: [0.718, 0.822, 0.926],
   336: [0.775, 0.911, 0.977],
   720: [0.939, 1.153, 1.218]},
  'Transformer': {96: [0.566, 0.707, 0.822],
   192: [0.704, 0.756, 0.883],
   336: [0.905, 0.976, 1.067],
   720: [0.903, 1.028, 1.076]},
  'iTransformer': {96: [0.321, 0.379, 0.412],
   192: [0.364, 0.419, 0.439],
   336: [0.411, 0.446, 0.483],
   720: [0.397, 0.431, 0.466]}},
 'ETTh2.csv': {'Autoformer': {96: [0.368, 0.434, 0.466],
   192: [0.368, 0.396, 0.522],
   336: [0.318, 0.399, 0.413],
   720: [0.47, 0.505, 0.543]},
  'FiLM': {96: [0.247, 0.31, 0.338],
   192: [0.318, 0.351, 0.421],
   336: [0.325, 0.429, 0.443],
   720: [0.401, 0.444, 0.48]},
  'Informer': {96: [1.109, 1.198, 1.398],
   192: [1.272, 1.502, 2.001],
   336: [1.397, 1.466, 1.57],
   720: [0.726, 0.883, 1.094]},
  'Nonstationary_Transformer': {96: [0.281, 0.371, 0.4],
   192: [0.472, 0.517, 0.638],
   336: [0.384, 0.523, 0.559],
   720: [0.413, 0.437, 0.492]},
  'Pyraformer': {96: [0.359, 0.766, 1.181],
   192: [1.117, 1.519, 3.002],
   336: [1.128, 1.631, 1.887],
   720: [0.595, 0.682, 0.772]},
  'Reformer': {96: [0.896, 1.695, 1.771],
   192: [0.656, 1.061, 1.262],
   336: [0.787, 1.1, 1.176],
   720: [1.282, 1.419, 1.603]},
  'Transformer': {96: [0.95, 1.058, 1.203],
   192: [0.565, 0.606, 0.635],
   336: [0.519, 0.665, 0.725],
   720: [0.527, 0.585, 0.65]},
  'iTransformer': {96: [0.239, 0.335, 0.373],
   192: [0.32, 0.394, 0.474],
   336: [0.352, 0.478, 0.503],
   720: [0.373, 0.4, 0.425]}},
 'ETTm1.csv': {'Autoformer': {96: [0.415, 0.438, 0.448],
   192: [0.473, 0.493, 0.547],
   336: [0.437, 0.458, 0.475],
   720: [0.536, 0.564, 0.579]},
  'FiLM': {96: [0.297, 0.318, 0.321],
   192: [0.313, 0.321, 0.364],
   336: [0.323, 0.367, 0.404],
   720: [0.379, 0.45, 0.469]},
  'Informer': {96: [0.522, 0.629, 0.701],
   192: [0.552, 0.573, 0.618],
   336: [0.584, 0.665, 0.759],
   720: [0.712, 0.782, 0.821]},
  'Nonstationary_Transformer': {96: [0.314, 0.351, 0.365],
   192: [0.35, 0.394, 0.433],
   336: [0.366, 0.452, 0.518],
   720: [0.468, 0.514, 0.555]},
  'Pyraformer': {96: [0.393, 0.427, 0.484],
   192: [0.409, 0.452, 0.474],
   336: [0.432, 0.468, 0.55],
   720: [0.503, 0.688, 0.744]},
  'Reformer': {96: [0.678, 0.744, 0.777],
   192: [0.686, 0.714, 0.732],
   336: [0.764, 0.802, 0.852],
   720: [0.851, 0.885, 0.946]},
  'Transformer': {96: [0.424, 0.434, 0.439],
   192: [0.478, 0.496, 0.571],
   336: [0.559, 0.595, 0.724],
   720: [0.737, 0.792, 0.816]},
  'iTransformer': {96: [0.284, 0.319, 0.323],
   192: [0.313, 0.323, 0.399],
   336: [0.315, 0.395, 0.45],
   720: [0.381, 0.486, 0.503]}},
 'ETTm2.csv': {'Autoformer': {96: [0.182, 0.228, 0.266],
   192: [0.223, 0.297, 0.351],
   336: [0.283, 0.383, 0.408],
   720: [0.397, 0.429, 0.473]},
  'FiLM': {96: [0.139, 0.177, 0.202],
   192: [0.18, 0.221, 0.316],
   336: [0.212, 0.33, 0.369],
   720: [0.345, 0.387, 0.437]},
  'Informer': {96: [0.186, 0.221, 0.248],
   192: [0.234, 0.304, 0.432],
   336: [0.319, 0.519, 0.89],
   720: [0.449, 0.904, 1.122]},
  'Nonstationary_Transformer': {96: [0.148, 0.19, 0.234],
   192: [0.191, 0.296, 0.376],
   336: [0.277, 0.369, 0.381],
   720: [0.402, 0.437, 0.456]},
  'Pyraformer': {96: [0.172, 0.202, 0.218],
   192: [0.242, 0.256, 0.341],
   336: [0.358, 0.486, 0.908],
   720: [0.399, 1.115, 1.538]},
  'Reformer': {96: [0.386, 0.495, 0.544],
   192: [0.563, 0.672, 0.762],
   336: [0.767, 0.914, 1.028],
   720: [1.016, 1.058, 1.111]},
  'Transformer': {96: [0.175, 0.217, 0.233],
   192: [0.272, 0.327, 0.443],
   336: [0.44, 0.563, 0.681],
   720: [0.498, 0.781, 1.037]},
  'iTransformer': {96: [0.127, 0.178, 0.21],
   192: [0.175, 0.234, 0.336],
   336: [0.226, 0.35, 0.397],
   720: [0.353, 0.421, 0.479]}},
 'electricity.csv': {'Autoformer': {96: [0.151, 0.16, 0.166],
   192: [0.174, 0.184, 0.202],
   336: [0.197, 0.207, 0.24],
   720: [0.218, 0.228, 0.244]},
  'FiLM': {96: [0.157, 0.203, 0.232],
   192: [0.169, 0.23, 0.235],
   336: [0.184, 0.214, 0.261],
   720: [0.185, 0.21, 0.274]},
  'Informer': {96: [0.224, 0.226, 0.232],
   192: [0.226, 0.237, 0.257],
   336: [0.25, 0.263, 0.296],
   720: [0.309, 0.321, 0.352]},
  'Nonstationary_Transformer': {96: [0.131, 0.143, 0.15],
   192: [0.141, 0.156, 0.176],
   336: [0.147, 0.174, 0.199],
   720: [0.182, 0.227, 0.255]},
  'Pyraformer': {96: [0.194, 0.197, 0.2],
   192: [0.208, 0.218, 0.234],
   336: [0.225, 0.235, 0.257],
   720: [0.245, 0.258, 0.277]},
  'Reformer': {96: [0.198, 0.202, 0.207],
   192: [0.218, 0.238, 0.254],
   336: [0.223, 0.236, 0.254],
   720: [0.251, 0.254, 0.255]},
  'Transformer': {96: [0.176, 0.181, 0.189],
   192: [0.197, 0.211, 0.227],
   336: [0.194, 0.211, 0.234],
   720: [0.232, 0.239, 0.25]},
  'iTransformer': {96: [0.12, 0.133, 0.137],
   192: [0.132, 0.138, 0.144],
   336: [0.135, 0.152, 0.182],
   720: [0.173, 0.207, 0.232]}},
 'exchange_rate.csv': {'Autoformer': {96: [0.138, 0.217, 0.269],
   192: [0.375, 0.742, 1.714],
   336: [0.763, 0.83, 1.126],
   720: [0.582, 1.047, 1.4]},
  'FiLM': {96: [0.178, 0.231, 0.283],
   192: [0.232, 0.353, 0.427],
   336: [0.679, 0.943, 1.203],
   720: [1.078, 1.83, 2.249]},
  'Informer': {96: [0.759, 0.8, 1.017],
   192: [0.826, 1.103, 1.332],
   336: [1.987, 2.337, 3.09],
   720: [3.706, 4.829, 6.314]},
  'Nonstationary_Transformer': {96: [0.11, 0.223, 0.337],
   192: [0.213, 0.385, 0.52],
   336: [0.273, 0.631, 1.01],
   720: [1.817, 2.356, 4.053]},
  'Pyraformer': {96: [0.925, 1.075, 1.292],
   192: [1.21, 1.476, 1.686],
   336: [2.067, 2.233, 2.374],
   720: [2.394, 3.794, 4.866]},
  'Reformer': {96: [1.137, 1.221, 1.495],
   192: [1.915, 2.067, 2.187],
   336: [3.682, 4.134, 4.544],
   720: [3.071, 4.638, 5.988]},
  'Transformer': {96: [0.423, 0.562, 0.67],
   192: [0.431, 0.798, 0.87],
   336: [1.019, 1.183, 1.294],
   720: [1.234, 2.496, 3.198]},
  'iTransformer': {96: [0.079, 0.13, 0.171],
   192: [0.134, 0.235, 0.29],
   336: [0.239, 0.405, 0.602],
   720: [0.487, 0.823, 1.107]}},
 'national_illness.csv': {'Autoformer': {24: [0.53, 0.734, 0.875],
   36: [0.271, 0.357, 0.481],
   48: [0.345, 0.422, 0.566],
   60: [0.295, 0.458, 0.617]},
  'FiLM': {24: [0.55, 0.628, 0.654],
   36: [0.342, 0.538, 0.964],
   48: [0.528, 0.609, 0.664],
   60: [0.32, 0.415, 0.844]},
  'Informer': {24: [0.173, 0.219, 0.275],
   36: [0.15, 0.258, 0.346],
   48: [0.129, 0.223, 0.361],
   60: [0.129, 0.222, 0.435]},
  'Nonstationary_Transformer': {24: [0.296, 0.374, 0.425],
   36: [0.242, 0.291, 0.332],
   48: [0.251, 0.344, 0.443],
   60: [0.209, 0.267, 0.602]},
  'Pyraformer': {24: [0.219, 0.257, 0.291],
   36: [0.185, 0.218, 0.24],
   48: [0.167, 0.208, 0.291],
   60: [0.114, 0.216, 0.317]},
  'Reformer': {24: [0.409, 0.461, 0.516],
   36: [0.303, 0.412, 0.522],
   48: [0.239, 0.293, 0.46],
   60: [0.201, 0.299, 0.659]},
  'Transformer': {24: [0.259, 0.283, 0.329],
   36: [0.296, 0.32, 0.341],
   48: [0.238, 0.375, 0.538],
   60: [0.21, 0.259, 0.731]},
  'iTransformer': {24: [0.222, 0.242, 0.259],
   36: [0.181, 0.217, 0.296],
   48: [0.164, 0.235, 0.32],
   60: [0.142, 0.195, 0.423]}},
 'traffic.csv': {'Autoformer': {96: [0.459, 0.464, 0.472],
   192: [0.457, 0.465, 0.483],
   336: [0.449, 0.466, 0.473],
   720: [0.432, 0.482, 0.558]},
  'FiLM': {96: [0.339, 0.346, 0.348],
   192: [0.332, 0.336, 0.341],
   336: [0.334, 0.363, 0.371],
   720: [0.367, 0.468, 0.502]},
  'Informer': {96: [0.567, 0.579, 0.599],
   192: [0.583, 0.603, 0.629],
   336: [0.62, 0.645, 0.678],
   720: [0.764, 0.82, 0.901]},
  'Nonstationary_Transformer': {96: [0.472, 0.491, 0.494],
   192: [0.444, 0.45, 0.459],
   336: [0.434, 0.455, 0.473],
   720: [0.417, 0.466, 0.536]},
  'Pyraformer': {96: [0.516, 0.532, 0.551],
   192: [0.534, 0.541, 0.548],
   336: [0.537, 0.558, 0.572],
   720: [0.485, 0.531, 0.608]},
  'Reformer': {96: [0.521, 0.535, 0.542],
   192: [0.531, 0.541, 0.55],
   336: [0.518, 0.545, 0.56],
   720: [0.468, 0.528, 0.609]},
  'Transformer': {96: [0.468, 0.474, 0.484],
   192: [0.476, 0.485, 0.493],
   336: [0.469, 0.502, 0.519],
   720: [0.415, 0.475, 0.567]},
  'iTransformer': {96: [0.35, 0.356, 0.376],
   192: [0.345, 0.354, 0.378],
   336: [0.343, 0.37, 0.396],
   720: [0.357, 0.411, 0.49]}},
 'weather.csv': {'Autoformer': {96: [0.484, 0.541, 0.549],
   192: [0.559, 0.58, 0.689],
   336: [0.61, 0.679, 0.737],
   720: [0.702, 0.862, 0.92]},
  'FiLM': {96: [0.473, 0.516, 0.524],
   192: [0.526, 0.547, 0.597],
   336: [0.554, 0.685, 0.702],
   720: [0.684, 0.797, 0.852]},
  'Informer': {96: [0.475, 0.527, 0.548],
   192: [0.525, 0.563, 0.607],
   336: [0.577, 0.671, 0.729],
   720: [0.747, 0.849, 0.98]},
  'Nonstationary_Transformer': {96: [0.404, 0.47, 0.504],
   192: [0.452, 0.527, 0.597],
   336: [0.562, 0.69, 0.798],
   720: [0.684, 0.868, 0.956]},
  'Pyraformer': {96: [0.412, 0.52, 0.535],
   192: [0.532, 0.576, 0.621],
   336: [0.581, 0.705, 0.762],
   720: [0.691, 0.808, 0.926]},
  'Reformer': {96: [0.461, 0.527, 0.565],
   192: [0.598, 0.627, 0.667],
   336: [0.73, 0.804, 0.835],
   720: [0.877, 0.892, 0.905]},
  'Transformer': {96: [0.467, 0.538, 0.585],
   192: [0.559, 0.662, 0.771],
   336: [0.622, 0.772, 0.869],
   720: [0.787, 0.972, 1.035]},
  'iTransformer': {96: [0.386, 0.472, 0.518],
   192: [0.478, 0.537, 0.559],
   336: [0.54, 0.623, 0.702],
   720: [0.629, 0.786, 0.845]}}}

CONSTRAINT_PARAMS={
  'constraint_level': {'values': []},#will fail if not set later.
  'constraint_type': {'value': CONSTRAINT_TYPE},
  'dual_init': {'value': DUAL_INIT},
  'dual_lr': {'value': DUAL_LR},
}
if RESILIENT_LR>0:
    CONSTRAINT_PARAMS['resilient_lr']={'value': RESILIENT_LR}
    CONSTRAINT_PARAMS['resilient_cost_alpha']={'value': 2.0}
else:
    CONSTRAINT_PARAMS['resilient_lr']={'value': 0.0}
    CONSTRAINT_PARAMS['resilient_cost_alpha']={'value': 0.0}
    
print("Constraint params (before adding levels): ")
print(CONSTRAINT_PARAMS)

SWEEP_HEADER={
  "program": "run.py",
  "method": "grid",
  "project": WANDB_PROJECT,
  "entity": NAMESPACE,
  "metric": {
    "name": "mse/test",
    "goal": "minimize"
  },
}

DOCUMENTATION_PARAMS = {
  'wandb_run': {'value': ''}, 
  'wandb_project': {'value': WANDB_PROJECT},
  'experiment_tag': {'value': EXPERIMENT_TAG},
  'des': {'value': EXPERIMENT_DESCRIPTION}, 
}

if MODELS == ["PatchTST"]:
    print("Running a PatchTST set of runs")
    PATCH_TST_PARAMS = {
        "fc_dropout": 0.2,
        "e_layers": 3, #Will overwrite the one in the template
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "lradj": "TST" 
    }
else: 
    PATCH_TST_PARAMS = {}

if MODELS == ["Koopa"]:
    print("Running a Koopa set of runs")
    KOOPA_PARAMS = {
        "dynamic_dim": {"value": 256},
        "hidden_dim": {"value": 512},
        "hidden_layers": {"value": 3},
        "seg_len": {"value": 48},
        "num_blocks": {"value": 3},
        "alpha": {"value": 0.2},
        "multistep": {"value": 'True'}
    }
else:
    KOOPA_PARAMS = {}

TEMPLATE={
    **SWEEP_HEADER,
    "parameters": {
        **DOCUMENTATION_PARAMS,
        #**DATASET_DEPENDENT, #will be added later
        "model": MODEL_DICT,
        'pred_len': {'value': 0}, #also should fail if not set earlier
        **CONSTRAINT_PARAMS,
        #Other params that don't change much
        'train_epochs': {'value': 10},
        'is_training': {'value': '1'},
        'seq_len': {'value': 96}, 
        'label_len': {'value': 48},
        'e_layers': {'value': 2},
        'd_layers': {'value': 1},
        'factor': {'value': 3},
        'itr': {'value': 1},
        'seed': {'value': SEED},
        **PATCH_TST_PARAMS,
        **KOOPA_PARAMS
    }
}
if CONSTRAINT_TYPE != 'erm':
  # Generating epsilon constraint sweeps
  # By definition, must run one at a time because the gridsearch is along constraint levels.
  # CONSRTTRAINED
  print(f"Starting sweep generation for constrained run of type {CONSTRAINT_TYPE}")
  sweep_ids = []
  for num_seed in range(1,NUM_SEEDS+1):
    for data_path in DATASETS:
      if data_path == 'national_illness.csv':
        PRED_LENGTHS = ILLNESS_PRED_LENGTHS
      else:
        PRED_LENGTHS = STANDARD_PRED_LENGTHS
      print(f"Generating sweep for {data_path} with pred lengths")
      print(PRED_LENGTHS)
      for model in MODELS:
          if model not in CONSTRAINT_DATA[data_path]:
            print(f"WARNING: Skipping {model} for {data_path} as no constraint data found")
            continue
          
          for pred_len in PRED_LENGTHS:
                if pred_len not in CONSTRAINT_DATA[data_path][model]:
                  print(f"WARNING: Skipping {model} for {data_path} and pred len {pred_len}, as no constraint data found")
                  continue
                sweep_config = TEMPLATE.copy()
                #data_path replace .csv with '', for names.
                data_path_nodot=data_path.replace('.csv','')

                sweep_config['name'] = f"{SWEEP_NAME_PREFIX}_{data_path_nodot}_{model}_{pred_len}_seed{num_seed}"
                sweep_config['parameters']['des'] = {'value': f"{EXPERIMENT_DESCRIPTION} {data_path} {model} {pred_len} seed{num_seed}."}
                sweep_config['parameters']['wandb_run'] = {'value': f"{data_path_nodot}/Constrained/{model}/{pred_len}-10e"}
                sweep_config["parameters"].update(DATASET_DEPENDENT[data_path])
                sweep_config["parameters"]["model"] = {"value": model}
                sweep_config["parameters"]["pred_len"] = {"value": pred_len}
                
                constraint_type_name = CONSTRAINT_PARAMS['constraint_type']['value']
                sweep_config["parameters"]["model_id"] = {"value": f"{model}_{data_path_nodot}_{pred_len}_{constraint_type_name}"}

                # Add the constraint levels
                sweep_config["parameters"].update(CONSTRAINT_PARAMS)
                
                if CONSTRAINT_TYPE=='static_linear':
                  #print("Adding static linear constraint levels")
                  sweep_config["parameters"]["constraint_slope"] = {"value": CONSTRAINT_LINES[data_path][model][pred_len]['constraint_slope']}
                  sweep_config["parameters"]["constraint_offset"] = {"value": CONSTRAINT_LINES[data_path][model][pred_len]['constraint_offset']}
                  sweep_config["parameters"].pop("constraint_level")
                #TODO refactor this to support both linear and monotonic. I'm too tired
                if CONSTRAINT_TYPE=='monotonic': # constraint_level-less
                  sweep_config["parameters"].pop("constraint_level")
                else:
                  if len(CONSTRAINT_DATA[data_path][model][pred_len]) == 1:
                    sweep_config["parameters"]["constraint_level"] = {"value": CONSTRAINT_DATA[data_path][model][pred_len][0]}  
                  elif len(CONSTRAINT_DATA[data_path][model][pred_len]) > 1: 
                    sweep_config["parameters"]["constraint_level"] = {"values": CONSTRAINT_DATA[data_path][model][pred_len]}
                  else:
                    raise ValueError("No constraint levels found for this model, dataset, and pred_len")
                #Update description, including params & seed number
                #print(sweep_config)
                sweep_id = wandb.sweep(sweep_config)
                sweep_ids.append(sweep_id)
                # Write YAML file for debugging, with overwrite
                YAML_FILENAME=f"sweep_{data_path_nodot}_{model}_{pred_len}_seed{num_seed}.yaml"
                #print(f"YAML can be debugged in {YAML_DEBUG_LOCATION+YAML_FILENAME}")
                with open(YAML_DEBUG_LOCATION+YAML_FILENAME,'w') as f:
                    yaml.dump(sweep_config,f,sort_keys=False)
elif CONSTRAINT_TYPE=='erm':
  #########
  ###########################
  ###########################
  ###########################
  ###FOR ERM
  #TODO maybe split into separate files or funcs. 
  # Generating ERM sweeps
  print(f"Starting sweep generation for ERM runs (constraint_type={CONSTRAINT_TYPE})")
  print("ERM Sweeps, setting constraints to zero")
  CONSTRAINT_PARAMS={
    'constraint_level': {'value': -1.0},
    'constraint_type': {'value': 'erm'},
    'dual_init': {'value': 0.0},
    'dual_lr': {'value': 0.0},
  }

  sweep_ids = []
  for num_seed in range(1,NUM_SEEDS+1):
    for data_path in DATASETS:
      if data_path == 'national_illness.csv':
        PRED_LENGTHS = ILLNESS_PRED_LENGTHS
      else:
        PRED_LENGTHS = STANDARD_PRED_LENGTHS
      sweep_config = TEMPLATE.copy()
      sweep_config['name'] = f"{SWEEP_NAME_PREFIX}_{data_path}_seed{SEED}"
      sweep_config['parameters']['des'] = {'value': f"{EXPERIMENT_DESCRIPTION} ERM epsilon run {data_path} seed{SEED}."}
      sweep_config['parameters']['wandb_run'] = {'value': f"AllModels_{data_path}/ERM"}
      sweep_config["parameters"].update(DATASET_DEPENDENT[data_path])
      sweep_config["parameters"]["model"] = {"values": MODELS}
      sweep_config["parameters"]["pred_len"] = {"values": PRED_LENGTHS}
      #data_path replace .csv with ''
      sweep_config["parameters"]["model_id"] = {"value": f"{data_path.replace('.csv','')}_erm"}

      # Add the placeholder constraint levels
      sweep_config["parameters"].update(CONSTRAINT_PARAMS)
      # drop constraint_levels
      #sweep_config["parameters"].pop("constraint_level")
      # Update description, including params & seed number
      #print(sweep_config)
      
      sweep_id = wandb.sweep(sweep_config)
      sweep_ids.append(sweep_id)
      # Write YAML file for debugging, with overwrite
      YAML_FILENAME=f"sweep_erm_{data_path}_all_models_seed{SEED}.yaml"
      #print(f"YAML can be debugged in {YAML_DEBUG_LOCATION+YAML_FILENAME}")
      with open(YAML_DEBUG_LOCATION+YAML_FILENAME,'w') as f:
          yaml.dump(sweep_config,f,sort_keys=False)
# PRINTING THE RESULTING GENERATED AGENT IDs
agents_array = reduce(lambda x, y: f"{x} {y}", sweep_ids)
sweep_command=f"""
agents=({agents_array})
for agent in "${{agents[@]}}"
do
  wandb agent "{NAMESPACE}/{WANDB_PROJECT}/$agent"
done
"""
print("Run the following commands: \n\n")
print(sweep_command)
  ###########################
  ###########################
  ###########################
  ###########################