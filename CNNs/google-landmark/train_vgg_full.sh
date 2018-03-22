DATASET_DIR=./data
TRAIN_DIR=./output/vgg_full
CHECKPOINT_PATH=./output/vgg_mlp

python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=15 \
    --learning_rate=0.00005 \
    --model_name=vgg_16 \
    --save_summaries_secs=50 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=224 \