export MODEL_NAME="sd-legacy/stable-diffusion-inpainting"
export DATA_DIR='datasets/mvtec_defect_collection/images'
CUDA_VISIBLE_DEVICES=0 accelerate launch train_global.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="elite_experiments/mvtec_global_mapping" \
  --save_steps 1000
