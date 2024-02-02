import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm

# Todo save in a file
def get_experiment_data(project,workspace,experiment_tags=[],state='finished',query_dict=None,timeout=60):
    """
    Downloads wandb data and returns a DataFrame with the relevant information from the experiment.
    Either choose experiment_tags  and state, or pass a full query_dict to filter the runs.
    """
    api = wandb.Api(timeout=timeout)
    
    if query_dict is None:
        query_dict={"$and": [
                        {"tags": {"$in": experiment_tags}},
                        {"state": state}
        ]}

    # get all runs that both: 1.  match any experiment tag and 2. are finished
    runs = api.runs(f"{workspace}/{project}",query_dict)

    def tag_experiment(run):
        for tag in experiment_tags:
            if tag in run.tags:
                return tag
        return ''

    # filter runs by 

    all_runs = []
    run_counter = 0
    for run in tqdm(runs):
        if "pointwise/cvar/0.01" not in run.summary:
            print(f"WARNING!! Missing pointwise/cvar/0.01 in run {run.id} on sweep {run.sweep.name}. Filling all poitwisewith NAN.")
        run_counter += 1
        for split in ["train", "test","val"]:
            for metric in ["mse"]:
                pred_len = run.config["pred_len"]
                metrics = np.zeros(pred_len)
                for i in range(pred_len):
                    run_dict = {**run.config}
                    #run_dict["constraint_level"] = constraint_level
                    run_dict[f"{metric}"] = run.summary[f"{metric}/{split}/{i}"]
                    #run_dict[f"{metric}"] = run.summary.get(f"{metric}/{split}/{i}",run.summary.get(f"mse/{split}/{i}",np.nan)) #god forgive me for this line
                    run_dict["step"]=i
                    run_dict["epoch"]=run.summary["epoch"]
                    run_dict["infeasible_rate"]=run.summary[f"infeasible_rate/{split}"]
                    run_dict["infeasibles"]=run.summary[f"infeasibles/{split}"]
                    run_dict[f"multiplier"] = run.summary[f"multiplier/{i}"] if split == "train" and f"multiplier/{i}" in run.summary else np.nan
                    
                    run_dict["split"] = split
                    run_dict["run_id"] = run.id
                    # Get either Constrained/ or ERM/ from the run name, then append model name.
                    #print("run.name", run.name)
                    #debug if ERM run
                    run_dict['run_name'] = run.name
                    run_dict["Algorithm"] = f"{run.name.split('/')[0]} {run.config['model']}"
                    sweep_id = run.sweep.id if run.sweep else np.nan
                    run_dict["sweep_id"] = sweep_id
                    run_dict["sweep_name"] = run.sweep.name if run.sweep else np.nan
                    #print("Algorithm", run_dict["Algorithm"])

                    # Get the experiment tag
                    run_dict["experiment_tag"] = tag_experiment(run)
                    
                    if run_dict["constraint_type"] == "resilience": #Rename pre-refactor runs for consistency.
                        run_dict["constraint_type"] = "constant_resilience"
                    else:
                        if "resilient_lr" in run.config and run.config['resilient_lr'] > 0:
                            run_dict["constraint_type"] = run_dict["constraint_type"] + "_resilience"
                    
                    run_dict[f"linearity/{split}"] = run.summary[f"linearity/{split}"]
                    run_dict["pct_50_total_test"] = run.summary["pct_50/test"]
                    run_dict["pct_50_total_val"] = run.summary["pct_50/val"]
                    run_dict["pct_75_total_test"] = run.summary["pct_75/test"]
                    run_dict["pct_75_total_val"] = run.summary["pct_75/val"]
                    run_dict["pct_95_total_test"] = run.summary["pct_95/test"]
                    run_dict["pct_95_total_val"] = run.summary["pct_95/val"]
                    run_dict["pct_99_total_test"] = run.summary["pct_99/test"]
                    run_dict["pct_99_total_val"] = run.summary["pct_99/val"]
                    
                    if split in ["test","val"]:
                        run_dict["pct_50"] = run.summary[f"pct_50_per_timestep/{split}/{i}"]
                        run_dict["pct_75"] = run.summary[f"pct_75_per_timestep/{split}/{i}"]
                        run_dict["pct_95"] = run.summary[f"pct_95_per_timestep/{split}/{i}"]
                        run_dict["pct_99"] = run.summary[f"pct_99_per_timestep/{split}/{i}"]
                    
                    if "pointwise/cvar/0.01" not in run.summary:
                        run_dict["pointwise/cvar/001"] = np.nan
                        run_dict["pointwise/cvar/005"] = np.nan
                        run_dict["pointwise/iqr"] = np.nan
                        run_dict["pointwise/max"] = np.nan
                        run_dict["pointwise/quantile/09"] = np.nan
                        run_dict["pointwise/quantile/095"] = np.nan
                        run_dict["pointwise/quantile/099"] = np.nan
                        run_dict["pointwise/std"] = np.nan
                    else:    
                        run_dict["pointwise/cvar/001"] = run.summary[f"pointwise/cvar/0.01"]
                        run_dict["pointwise/cvar/005"] = run.summary["pointwise/cvar/0.05"]
                        run_dict["pointwise/iqr"] = run.summary["pointwise/iqr"]
                        run_dict["pointwise/max"] = run.summary["pointwise/max"]
                        run_dict["pointwise/quantile/09"] = run.summary["pointwise/quantile/0.9"]
                        run_dict["pointwise/quantile/095"] = run.summary["pointwise/quantile/0.95"]
                        run_dict["pointwise/quantile/099"] = run.summary["pointwise/quantile/0.99"]
                        run_dict["pointwise/std"] = run.summary["pointwise/std"]
                    
                    # To better plot constrained vs ERM
                    #TODO this is a hack while I consolidate the tags. 
                    run_dict["type"] = "ERM" if run.config['dual_lr'] == 0 else "Constrained"

                    all_runs.append(run_dict)
    print(f"Fetched {run_counter} runs")
    df = pd.DataFrame(all_runs)
    print(f"Total records: {(df.shape)}")
    print(f"Total runs: {df.run_id.nunique()}")
    return df

def generate_constraint_levels(df,split='train'):
    """
    Generate the constraint level dictionary from ERM runs. 

    USE THIS FUNCTION TO POPULATE THE CONSTRAINT LEVELS ON CONSTANT ERM RUNS.

    Expected format: 
    {
        'data_path': {
            'model': {
                'pred_len': [25%,50%,75%]
            }
        }
    }
    
    """
    erm_df=df.query(f'constraint_type=="erm" and split=="{split}"').copy()

    # Get only a single run per (data_path,model,pred_len) to compute results
    erm_df_run_ids = erm_df[['data_path','model','pred_len','run_id']].drop_duplicates(['data_path','model','pred_len']).run_id.tolist()
    print("Generating constraints based on the following runs (one per param set)")
    print(erm_df_run_ids)
    filtered_erm_df = erm_df[erm_df.run_id.isin(erm_df_run_ids)].copy()

    stats=filtered_erm_df.groupby(['data_path','model','pred_len'])['mse'].describe().reset_index()

    constraint_data=stats[['data_path','model','pred_len','25%','50%','75%','mean','std']].sort_values(['data_path','pred_len','model'])
    
    models = constraint_data.model.unique()
    pred_lens = constraint_data.pred_len.unique()
    data_paths = constraint_data.data_path.unique()
    constraint_data_dict = {}
    for data_path in data_paths:
        constraint_data_dict[data_path]={}
        for model in models:
            constraint_data_dict[data_path][model]={}
            for pred_len in pred_lens:                
                constraint_data_dict[data_path][model][pred_len]=constraint_data[(constraint_data.model==model) & (constraint_data.pred_len==pred_len) & (constraint_data.data_path==data_path)][['25%','50%','75%']].values.tolist()[0]
                # Round to 3 decimals
                constraint_data_dict[data_path][model][pred_len] = [round(x,3) for x in constraint_data_dict[data_path][model][pred_len]]
    return constraint_data_dict