export CUDA_VISIBLE_DEVICES=6

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/traffic/ \
  --data-path traffic.csv \
  --model-id traffic_96_96 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 862 \
  --dec-in 862 \
  --c-out 862 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 3

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/traffic/ \
  --data-path traffic.csv \
  --model-id traffic_96_192 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 862 \
  --dec-in 862 \
  --c-out 862 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 3

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/traffic/ \
  --data-path traffic.csv \
  --model-id traffic_96_336 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 862 \
  --dec-in 862 \
  --c-out 862 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 3

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/traffic/ \
  --data-path traffic.csv \
  --model-id traffic_96_720 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 862 \
  --dec-in 862 \
  --c-out 862 \
  --des 'Exp' \
  --itr 1 \
  --train-epochs 3