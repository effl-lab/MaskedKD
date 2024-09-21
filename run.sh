python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 256 --data-path ../imagenet --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./results | tee log_mask50.txt

# python main.py --no_distributed \
# --model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
# --batch-size 1024 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee log_mask50.txt

# python3 run_timm_example.py \
#     --model_name_or_path "timm/fastvit_t8.apple_in1k" \
#     --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" \
#     --warmup 3 \
#     --n_iterations 5000 \
#     --use_hpu_graphs \
#     --bf16 \
#     --print_result

# python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
# --model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
# --batch-size 128 --data-path ./dataset --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./

