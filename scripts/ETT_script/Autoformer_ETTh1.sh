export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/ETT-small/ \
  --data-path ETTh1.csv \
  --model-id ETTh1_96_24 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
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
  --root-path ./dataset/ETT-small/ \
  --data-path ETTh1.csv \
  --model-id ETTh1_96_48 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
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
  --root-path ./dataset/ETT-small/ \
  --data-path ETTh1.csv \
  --model-id ETTh1_96_168 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 168 \
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
  --root-path ./dataset/ETT-small/ \
  --data-path ETTh1.csv \
  --model-id ETTh1_96_336 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
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
  --root-path ./dataset/ETT-small/ \
  --data-path ETTh1.csv \
  --model-id ETTh1_96_720 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 7 \
  --dec-in 7 \
  --c-out 7 \
  --des 'Exp' \
  --itr 1