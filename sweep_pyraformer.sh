agents=(po97at8s 7a8yrbli l2z9fxye n62an541 qy7iof9f npxbz43x ahpwclj2 o91a6jis 332p95ix kdrj60ez 522zkto5 66bweefj ui2i26ig vzv6c4he 5uhjpnsy rj0sv3jc)
for agent in "${agents[@]}"
do
CUDA_VISIBLE_DEVICES=0 wandb agent "hounie/Autoformer/$agent"
done