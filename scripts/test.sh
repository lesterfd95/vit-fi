#name pattern 'vit-all-ber-1e-12'
name=${1:-"vit-noname-last"}
model=${2:-"vit-noname-last"}

# Lista de valores BER
lista_ber=("1e-11" "1e-10" "1e-09" "1e-08" "2e-08" "4e-08" "6.5e-08" "1e-07" "3.16e-07" "1e-06")

# Lista de configuraciones de capas
layers_list=(
    "Linear"
    "LayerNorm"
    "GELU"
    "Dropout"
    "Softmax"
    "GELU Softmax"
    "Linear LayerNorm Dropout"
    "LayerNorm GELU Softmax"
    "Linear Dropout"
)

# Bucle externo sobre capas
for layers_config in "${layers_list[@]}"; do

    # Bucle interno sobre BER
    for ber in "${lista_ber[@]}"; do
        python3 tt_inj.py \
        --name $name \
        --lr 1e-4 \
        --opt "adam" \
        --noamp \
        --bs 100 \
        --size "224" \
        --n_epochs '100' \
        --dimhead "768" \
        --patch 16 \
        --layers_t Linear LayerNorm GELU Dropout Softmax \
        --shape_t 3 224 224 \
        --ber_t 1e-9 \
        --batchsize 200 \
        --shape 3 224 224 \
        --ber $ber \
        --layers $layers_config \
        --model $model \
        --seed 12 2>&1 | tee -a log/full_$name.log
    done
done



