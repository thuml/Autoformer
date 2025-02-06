from utils import wandb_utils

def backup_experiment_data(experiment_tags, filename_prefix):
        """
        Experiment Utility that queries W&B for finished experiments matching the provided tags, downloads the data,
        and saves to CSV, pickle.
        
        Args:
            experiment_tags (list): List of tags to filter W&B experiments
            filename_prefix (str): Prefix for the output filenames
            
        Returns:
            pandas.DataFrame: The retrieved experiment data as a pandas DataFrame
        """
        backup_data_filename = filename_prefix
        backup_data_pickle_filename = filename_prefix + ".pkl"
        backup_data_csv_filename = filename_prefix + ".csv"

        print("Experiment tags being considered:", experiment_tags)
        print("Saving to backup data filename:", backup_data_filename)

        backup_data = wandb_utils.get_experiment_data("Autoformer", "alelab", experiment_tags=experiment_tags, query_dict={"$and": [
                {"tags": {"$in": experiment_tags}},
                {"state": "finished"},
                #{"config.seed": 2021}
        ]})

        backup_data.to_pickle(backup_data_pickle_filename)
        backup_data.to_csv(backup_data_csv_filename, index=False)

        print("Backup data saved as pickle:", backup_data_pickle_filename)
        print("Backup data saved as CSV:", backup_data_csv_filename)
        print("Returning backed up data as a pandas DF")
        return backup_data
