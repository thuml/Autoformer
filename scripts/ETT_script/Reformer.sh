export CUDA_VISIBLE_DEVICES=0

# multivariate setting of vanilla transformer
python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm1.csv \
  --model-id ETTm1_96_96 \
  --model Reformer \
  --data ETTm1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --freq 't' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm1.csv \
  --model-id ETTm1_96_192 \
  --model Reformer \
  --data ETTm1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --freq 't' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm1.csv \
  --model-id ETTm1_96_336 \
  --model Reformer \
  --data ETTm1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --freq 't' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm1.csv \
  --model-id ETTm1_96_720 \
  --model Reformer \
  --data ETTm1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --freq 't' \
  --itr 1