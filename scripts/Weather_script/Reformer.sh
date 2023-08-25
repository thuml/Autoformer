export CUDA_VISIBLE_DEVICES=7


python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_96 \
  --model Reformer \
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
  --des 'Exp Javier' \
  --itr 1 \
  --train-epochs 3

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_192 \
  --model Reformer \
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
  --des 'Exp Javier' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_336 \
  --model Reformer \
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
  --des 'Exp Javier' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/weather/ \
  --data-path weather.csv \
  --model-id weather_96_720 \
  --model Reformer \
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
  --des 'Exp Javier' \
  --itr 1