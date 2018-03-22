DATASET_DIR=./data
TRAIN_DIR=./output/vgg_mlp
CHECKPOINT_PATH=./vgg_16.ckpt

python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=15 \
    --learning_rate=0.0001 \
    --model_name=vgg_16 \
    --save_summaries_secs=50 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=224 \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8
