export CUDA_VISIBLE_DEVICES=7

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/illness/ \
  --data-path national_illness.csv \
  --model-id ili_36_24 \
  --model Informer \
  --data custom \
  --features M \
  --seq-len 36 \
  --label-len 18 \
  --pred-len 24 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/illness/ \
  --data-path national_illness.csv \
  --model-id ili_36_36 \
  --model Informer \
  --data custom \
  --features M \
  --seq-len 36 \
  --label-len 18 \
  --pred-len 36 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/illness/ \
  --data-path national_illness.csv \
  --model-id ili_36_48 \
  --model Informer \
  --data custom \
  --features M \
  --seq-len 36 \
  --label-len 18 \
  --pred-len 48 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/illness/ \
  --data-path national_illness.csv \
  --model-id ili_36_60 \
  --model Informer \
  --data custom \
  --features M \
  --seq-len 36 \
  --label-len 18 \
  --pred-len 60 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --itr 1