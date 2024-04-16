agents=(hp4bdxh3 46ljfq91 npbyznzo lad0tqn0 kxlj1veh ikfi4uft zdeoboxq r3fgxcnq ent3wzad 1kun4lq2 gdz5nlsl xjew1otl)
for agent in "${agents[@]}"
do
CUDA_VISIBLE_DEVICES=1 wandb agent "hounie/Autoformer/$agent"
done