agents=(q7w0zqg6 e1mphf5w t1lnwrot ztuzao6h zer1rip1 2vmw81q3 g04klqp9 a2qyslz6 3birx81l x18lljt3 ed4fjcub 2ty5pn1m bzw2ch1v rerymdb5 82fu5o10 2e2lqla8)
for agent in "${agents[@]}"
do
CUDA_VISIBLE_DEVICES=1 wandb agent "hounie/Autoformer/$agent"
done