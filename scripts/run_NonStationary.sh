model="Nonstationary_Transformer"
for folder in "ECL" "Exchange" "Traffic" "Weather"
do
    CUDA_VISIBLE_DEVICES=0 ./scripts/${folder}_script/${model}.sh 
done