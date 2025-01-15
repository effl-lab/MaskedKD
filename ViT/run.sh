# imagenet21k pre-train
# wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# # imagenet21k pre-train + imagenet2012 fine-tuning
# wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz

## Running MaskedKD with 50% ratio 

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name B2S_maskedkd_98 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 200000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-B_16 --teacher_pretrained_dir model/path/ViT-B_16.pth \
--maskedkd --len_num_keep 98 \
--log_path logs/B2S_maskedkd_98 | tee running_logs/B2S_maskedkd_98.txt