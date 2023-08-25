export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm2.csv \
  --model-id ETTm2_96_96 \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq-len 96 \
  --label-len 96 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --freq 't' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm2.csv \
  --model-id ETTm2_96_192 \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq-len 96 \
  --label-len 96 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --itr 1 \
  --freq 't' \
  --train-epochs 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm2.csv \
  --model-id ETTm2_96_336 \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq-len 96 \
  --label-len 96 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --freq 't' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTm2.csv \
  --model-id ETTm2_96_720 \
  --model Autoformer \
  --data ETTm2 \
  --features S \
  --seq-len 96 \
  --label-len 96 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --freq 't' \
  --itr 1
