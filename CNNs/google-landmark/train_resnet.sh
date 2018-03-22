DATASET_DIR=./data/
TRAIN_DIR=./output_part_pretr_resnet/
CHECKPOINT_PATH=./checkpoints_resnet/resnet_v2_50.ckpt

python train_top5.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=50 \
    --learning_rate=0.0001 \
    --model_name=resnet_v2_50 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=180 \
    --save_summaries_secs=100 \
    --save_interval_secs=100 \
    --checkpoint_exclude_scopes=resnet_v2_50/logits
