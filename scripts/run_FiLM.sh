model="FiLM"
for folder in "Traffic" "Weather"
do
    CUDA_VISIBLE_DEVICES=0 ./scripts/${folder}_script/${model}.sh 
done