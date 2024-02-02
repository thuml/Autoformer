from utils import wandb_utils

experiment_tags = ['e16_finaljan_allerm_replicated',
'e17_finaljan_constant_constrained_replicated',
'e18_icml_constant_constrained_loose',
'e19_icml_resilience_val',
'e19_icml_resilience_train',
'e20_icml_monotonic',
'e20_icml_monotonic_no_resilience',
'e21_icml_static_linear_no_resilience',
'e21_icml_static_linear_resilience',]

backup_data=wandb_utils.get_experiment_data("Autoformer","alelab",experiment_tags=experiment_tags,query_dict={"$and": [
                        {"tags": {"$in": experiment_tags}},
                        {"state": "finished"},
                        {"config.seed": 2021}
        ]})
backup_data.to_pickle("data_backups/2024_02_01_1038am__icml_backup_data.pkl")
backup_data.to_csv("data_backups/2024_02_01_1038am__icml_backup_data.csv",index=False)