DATASET_DIR=./data
TRAIN_DIR=./output/resnet 
CHECKPOINT_PATH=./resnet_v2_50.ckpt

python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=100 \
    --learning_rate=0.0001 \
    --model_name=resnet_v2_50 \
    --save_summaries_secs=50 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=180 \
    --checkpoint_exclude_scopes=resnet_v2_50/logits\
    --trainable_scopes=resnet_v2_50/logits \