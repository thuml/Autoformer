export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/electricity/ \
  --data-path electricity.csv \
  --model-id ECL_96_96 \
  --model Transformer \
  --data custom \
  --features S \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 96 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/electricity/ \
  --data-path electricity.csv \
  --model-id ECL_96_192 \
  --model Transformer \
  --data custom \
  --features S \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 192 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/electricity/ \
  --data-path electricity.csv \
  --model-id ECL_96_336 \
  --model Transformer \
  --data custom \
  --features S \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 336 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is-training 1 \
  --root-path ./dataset/electricity/ \
  --data-path electricity.csv \
  --model-id ECL_96_720 \
  --model Transformer \
  --data custom \
  --features S \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 720 \
  --e-layers 2 \
  --d-layers 1 \
  --factor 3 \
  --enc-in 1 \
  --dec-in 1 \
  --c-out 1 \
  --des 'Exp' \
  --itr 1