# Loss Shaping Constraints for Time Series Forecasting

Official implementation of the experiments in ["Loss Shaping Constraints for Long-Term Time Series Forecasting"](https://arxiv.org/abs/2402.09373) (ICML 2024). Forked from [Autoformer](https://github.com/thuml/Autoformer).

## Setup
Environment setup:

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate loss-shaping
```

Set up W&B (optional):

```bash
wandb login
```

The datasets can be downloaded from [this link](https://drive.google.com/file/d/1uloXiB57ofJKA7Ozayy4wVLNuYv8Tbbq/view?usp=sharing). Alternatively, run `make get_dataset`. The expected structure is `{root_path}/{data_path}.csv`.

## Usage
The following examples show how to run ERM and constrained trainings using the Autoformer model on the ECL dataset.

To avoid using W&B, you can add the flag `WANDB_DISABLED=true` to the python command, or just use the CLI prompt when running the script.

### ERM Training
```bash
python run.py \
    --wandb_project WandbProject \
    --wandb_run example-run \
    --model Autoformer \
    --model_id electricity \
    --data custom \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --train_epochs 25 \
    --patience 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --is_training 1 \
    --constraint_type erm
```

### Constrained Training
```bash
python run.py \
    --wandb_project WandbProject \
    --wandb_run example-run \
    --model Autoformer \
    --model_id electricity \
    --data custom \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --train_epochs 25 \
    --patience 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --is_training 1 \
    --constraint_type constant \
    --constraint_level 0.167 \
    --dual_lr 0.01 \
    --dual_init 1.0
```

Key arguments (added by us):
- `constraint_type`: Type of constraint to use during training. `ERM` is unconstrained. 
  - Supported constraint types: `erm, constant, static_linear, dynamic_linear, resilience, monotonic, static_exponential`
- `constraint_level`: Upper bound on loss at each timestep (for constant constraints)
- `dual_lr`: Learning rate for dual variables
- `dual_init`: Initial value for dual variables.

Alternatively, the `sweeps/` directory contains YAML files for running W&B Sweeps, which were used for the paper experiments. To run: 

```bash
wandb sweep sweeps/{sweep_name}.yaml
wandb agent {sweep_id} #get the sweep_id from the previous command's output.
```

For more details on other arguments see the argparse help, or refer to the original [Autoformer](https://github.com/thuml/Autoformer) repository.

## Supported Models

**Transformer-Based:**
- Autoformer
- Informer
- Reformer
- Pyraformer
- iTransformer
- Nonstationary Transformer
- Vanilla Transformer

**Other:**
- FiLM (Feature-wise Linear Modulation)

## Supported Datasets
- Electricity Consumption Load (ECL)
- ETT (Electricity Transformer Temperature)
- Weather
- Exchange Rate
- Traffic
- Influenza-Like Illness (ILI)

## Acknowledgements

This implementation builds upon the [Autoformer](https://github.com/thuml/Autoformer) repository. We thank the original authors for their open-source contribution.