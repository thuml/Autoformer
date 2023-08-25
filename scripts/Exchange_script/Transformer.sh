export CUDA_VISIBLE_DEVICES=4


python -u run.py \
  --is-training 1 \
  --root-path ./dataset/exchange_rate/ \
  --data-path exchange_rate.csv \
  --model-id Exchange_96_96 \
  --model Transformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 8 \
  --dec-in 8 \
  --c-out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/exchange_rate/ \
  --data-path exchange_rate.csv \
  --model-id Exchange_96_192 \
  --model Transformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 8 \
  --dec-in 8 \
  --c-out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/exchange_rate/ \
  --data-path exchange_rate.csv \
  --model-id Exchange_96_336 \
  --model Transformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 8 \
  --dec-in 8 \
  --c-out 8 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/exchange_rate/ \
  --data-path exchange_rate.csv \
  --model-id Exchange_96_720 \
  --model Transformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 8 \
  --dec-in 8 \
  --c-out 8 \
  --des 'Exp' \
  --itr 1