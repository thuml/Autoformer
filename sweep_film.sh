agents=(babadza6 4uvzloog 4kln71c8 nxmoyvtw 4vy9rft3 2rdl0odq fmgums4f gjtbbnn1)
for agent in "${agents[@]}"
do
CUDA_VISIBLE_DEVICES=0 wandb agent "hounie/Autoformer/$agent"
done