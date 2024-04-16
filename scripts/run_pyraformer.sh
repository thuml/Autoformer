model="Pyraformer"
for folder in "ECL" "Exchange" "Traffic" "Weather"
do
    CUDA_VISIBLE_DEVICES=1 ./scripts/${folder}_script/${model}.sh 
done