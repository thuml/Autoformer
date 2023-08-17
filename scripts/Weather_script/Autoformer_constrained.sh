export CUDA_VISIBLE_DEVICES=0
for dual_init in 1.0 0.1
do
  for dual_lr in 0.01 0.1
  do
    for epsilon in 0.5 0.45 0.55
    do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_96 \
      --model Autoformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 2 \
      --dual_lr $dual_lr \
      --constraint_level $epsilon \
      --dual_init $dual_init \
      --wandb_run 'Constrained'

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_192 \
      --model Autoformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 192 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --dual_lr $dual_lr \
      --constraint_level $epsilon \
      --dual_init $dual_init \
      --wandb_run 'Constrained'

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_336 \
      --model Autoformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 336 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --dual_lr $dual_lr \
      --constraint_level $epsilon \
      --dual_init $dual_init \
      --wandb_run 'Constrained'

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_720 \
      --model Autoformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --dual_lr $dual_lr \
      --constraint_level $epsilon \
      --dual_init $dual_init \
      --wandb_run 'Constrained'
    done
  done
done
