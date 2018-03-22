DATASET_DIR=./data/
TRAIN_DIR=./output_part_pretr_inception/
CHECKPOINT_PATH=./checkpoints_inception/inception_v2.ckpt

python train_top5.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=50 \
    --learning_rate=0.0001 \
    --model_name=inception_v2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=180 \
    --save_summaries_secs=100 \
    --save_interval_secs=100 \
    --checkpoint_exclude_scopes=inception_v2/Logits
