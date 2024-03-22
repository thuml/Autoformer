model="Pyraformer"
for folder in "ECL" "Exchange" "Traffic" "Weather"
do
    ./scripts/${folder}_script/${model}.sh 
done