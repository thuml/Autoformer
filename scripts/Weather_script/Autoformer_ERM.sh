export CUDA_VISIBLE_DEVICES=0
dual_lr=0
WANDB_PROJECT='Autoformer-dev'

# WANDB_RUN_NAME='ERM'
WANDB_RUN_NAME='ERM-newloss'

echo "Triggering run $WANDB_RUN_NAME on wandb project $WANDB_PROJECT"

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_96 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 21 \
  --dec-in 21 \
  --c-out 21 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 2 \
  --dual-lr $dual_lr \
  --wandb-run $WANDB_RUN_NAME \
  --wandb-project $WANDB_PROJECT

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_192 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 21 \
  --dec-in 21 \
  --c-out 21 \
  --des 'Exp' \
  --itr 1 \
  --dual-lr $dual_lr \
  --wandb-run $WANDB_RUN_NAME \
  --wandb-project $WANDB_PROJECT

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_336 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 21 \
  --dec-in 21 \
  --c-out 21 \
  --des 'Exp' \
  --itr 1 \
  --dual-lr $dual_lr \
  --wandb-run $WANDB_RUN_NAME \
  --wandb-project $WANDB_PROJECT

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_720 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 21 \
  --dec-in 21 \
  --c-out 21 \
  --des 'Exp' \
  --itr 1 \
  --dual-lr $dual_lr \
  --wandb-run $WANDB_RUN_NAME \
  --wandb-project $WANDB_PROJECT