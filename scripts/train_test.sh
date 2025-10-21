#name pattern 'vit-all-ber-1e-12'
name=${1:-"vit-noname-last"}
model=${2:-"vit-noname-last"}

python3 tt_inj.py \
--resume \
--train \
--name $name \
--lr 1e-4 \
--opt "adam" \
--noamp \
--bs 256 \
--size "224" \
--n_epochs '100' \
--dimhead "768" \
--patch 16 \
--layers_t Linear LayerNorm\
--shape_t 3 224 224 \
--ber_t 6.5e-8 \
--batchsize 256 \
--shape 3 224 224 \
--ber 6.5e-8 \
--layers Linear LayerNorm GELU Dropout Softmax \
--model $model \
--seed 25 2>&1 | tee -a log/full_$name.log