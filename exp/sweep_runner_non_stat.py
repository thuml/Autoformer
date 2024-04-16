"""
Script to run sweeps 2.0. Before, we used YAMLs. Now we use the wandb API on python.
"""
import yaml
import os
import wandb

###### SCRIPT PARAMETERS

WANDB_PROJECT="Autoformer"
#WANDB_PROJECT="Autoformer-javierdev"
NAMESPACE="alelab"
YAML_DEBUG_LOCATION="../generated_sweeps/"
if not os.path.exists(YAML_DEBUG_LOCATION):
    os.makedirs(YAML_DEBUG_LOCATION)

SWEEP_NAME_PREFIX="ICML_ERM_NEWMODELS"
EXPERIMENT_TAG="e29_icml_constrained_newmodels"
EXPERIMENT_DESCRIPTION='Constrained New Models'

# Constraint parameters
# Constant
#CONSTRAINT_TYPE='resilience'
#CONSTRAINT_TYPE='constant'
#CONSTRAINT_TYPE='monotonic'
#CONSTRAINT_TYPE='static_linear'

DUAL_LR=0.01
DUAL_INIT=1.0

CONSTRAINT_TYPE='constant'


# Required if Resilience
# RESILIENT_LR=0.1
# Use if monotonic_no_resilience
RESILIENT_LR=0.0

#PROD PARAMETERS
MODELS = ["Nonstationary_Transformer"]#["Autoformer","Reformer","Informer","Transformer"]
#MODELS = ["PatchTST"] #Must run by itself
# MODELS = ["Koopa"]
if len(MODELS)>1 and "PatchTST" in MODELS:
    raise ValueError("PatchtTST Must be run separately because of its unique parameters")
DATASETS=[
  "electricity.csv","exchange_rate.csv","traffic.csv"]#"weather.csv",
  #"ETTh1.csv","ETTh2.csv","ETTm1.csv","ETTm2.csv"]
PRED_LENGTHS = [96,192,336,720]

# DATASETS=["illness.csv"] # Must be run on its own
#PRED_LENGTHS = [24, 36, 48, 60] # For ILI dataset
NUM_SEEDS=1
# SEED=0 # IF 0 THEN RANDOM SEED
SEED=2021 # The seed used by the literature
#SEED=6163 # Seed 2.
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
}

#train
CONSTRAINT_DATA={'electricity.csv': {'Nonstationary_Transformer': {96: [0.131, 0.143, 0.15],
   192: [0.141, 0.156, 0.176],
   336: [0.147, 0.174, 0.199],
   720: [0.182, 0.227, 0.255]}},
 'exchange_rate.csv': {'Nonstationary_Transformer': {96: [0.11, 0.223, 0.337],
   192: [0.213, 0.385, 0.52],
   336: [0.273, 0.631, 1.01],
   720: [1.817, 2.356, 4.053]}},
 'traffic.csv': {'Nonstationary_Transformer': {96: [0.472, 0.491, 0.494],
   192: [0.444, 0.45, 0.459],
   336: [0.434, 0.455, 0.473],
   720: [0.417, 0.466, 0.536]}}}

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
        'd_ff': {'value': 512},
        'd_model': {'value': 256},
        'top_k': {'value': 5},
        **PATCH_TST_PARAMS,
        **KOOPA_PARAMS
    }
}
if CONSTRAINT_TYPE is not 'erm':
  # Generating epsilon constraint sweeps
  # By definition, must run one at a time because the gridsearch is along constraint levels.
  # CONSRTTRAINED
  print(f"Starting sweep generation for constrained run of type {CONSTRAINT_TYPE}")
  sweep_commands = []
  for num_seed in range(1,NUM_SEEDS+1):
    for data_path in DATASETS:
      for model in MODELS:
          for pred_len in PRED_LENGTHS:
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
                sweep_commands.append(sweep_id)
                # Write YAML file for debugging, with overwrite
                YAML_FILENAME=f"sweep_{data_path_nodot}_{model}_{pred_len}_seed{num_seed}.yaml"
                #print(f"YAML can be debugged in {YAML_DEBUG_LOCATION+YAML_FILENAME}")
                with open(YAML_DEBUG_LOCATION+YAML_FILENAME,'w') as f:
                    yaml.dump(sweep_config,f,sort_keys=False)
  print("Run the following commands: \n\n")
  #print(sweep_commands)
  from functools import reduce
  # result = reduce(lambda x, y: f"{x} && {y}", sweep_commands)
  # print(result)
  # print ("\n Done")
  agents_array = reduce(lambda x, y: f"{x} {y}", sweep_commands)
  sweep_command=f"""
  agents=({agents_array})
  for agent in "${{agents[@]}}"
  do
    wandb agent "{NAMESPACE}/{WANDB_PROJECT}/$agent"
  done
  """
  print(sweep_command)
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

  sweep_commands = []
  for num_seed in range(1,NUM_SEEDS+1):
    for data_path in DATASETS:
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
      sweep_commands.append(f"wandb agent {NAMESPACE}/{WANDB_PROJECT}/{sweep_id}")
      # Write YAML file for debugging, with overwrite
      YAML_FILENAME=f"sweep_erm_{data_path}_all_models_seed{SEED}.yaml"
      #print(f"YAML can be debugged in {YAML_DEBUG_LOCATION+YAML_FILENAME}")
      with open(YAML_DEBUG_LOCATION+YAML_FILENAME,'w') as f:
          yaml.dump(sweep_config,f,sort_keys=False)
  from functools import reduce
  result = reduce(lambda x, y: f"{x} && {y}", sweep_commands)
  print(result)
  print ("\n Done")
  ###########################
  ###########################
  ###########################
  ###########################