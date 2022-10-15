import wandb

def generate_wandb_config(training_params):
    wandb.init(project=training_params.wandb_project_name)
    config = wandb.config
    config.model = training_params.model
    config.model_id = training_params.model_id
    config.is_training = training_params.is_training
    config.learning_rate = training_params.learning_rate
    config.loss = training_params.loss
    config.train_epochs = training_params.train_epochs
    config.batch_size = training_params.batch_size
    config.lr_adj = training_params.lradj
    config.dataset = training_params.data_path
    wandb_config = {
        'params': config,
        'project': training_params.wandb_project_name
    } 
    return wandb_config