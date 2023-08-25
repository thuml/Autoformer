export CUDA_VISIBLE_DEVICES=0
WANDB_PROJECT='Autoformer-javierdev'

#WANDB_RUN_NAME='Constrained'
WANDB_RUN_NAME='Constrained-newloss'

echo "Triggering run $WANDB_RUN_NAME on wandb project $WANDB_PROJECT"

for dual_init in 1.0 0.1
do
  for dual_lr in 0.01 0.1
  do
    #for epsilon in 0.5 0.45 0.55
    #For now we're doing just the one that works best on weather.
    for epsilon in 0.45
    do
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
      --constraint-level $epsilon \
      --dual-init $dual_init \
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
      --constraint-level $epsilon \
      --dual-init $dual_init \
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
      --constraint-level $epsilon \
      --dual-init $dual_init \
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
      --constraint-level $epsilon \
      --dual-init $dual_init \
      --wandb-run $WANDB_RUN_NAME \
      --wandb-project $WANDB_PROJECT
    done
  done
done
