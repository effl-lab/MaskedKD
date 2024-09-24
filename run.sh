mpirun -n 4 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u main.py \
--no_distill  --is_autocast --use_hpu --run_lazy_mode \
--model deit_tiny_patch16_224 --teacher_model deit3_small --epochs 2 \
--batch-size 256 --data-path ../imagenet --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./results | tee -a log_.txt

# $PYTHON -u train.py --name imagenet1k_TF --dataset imagenet1K --data_path /data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 64 --gradient_accumulation_steps 2 --img_size 384 --learning_rate 0.06 --autocast

# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
# --no_distill --hf_model \
# --model google/vit-base-patch16-224-in21k  --teacher_model deit3_small --epochs 1 \
# --batch-size 256 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee -a log_test.txt

# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
# --model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 1 \
# --batch-size 256 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee -a log_test.txt


# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
# --model vit_small_patch16_224  --teacher_model deit3_small --epochs 300 \
# --batch-size 256 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee log_mask50.txt

# python main.py --no_distributed \
# --no_distill \
# --model google/vit-base-patch16-224-in21k  --teacher_model deit3_small --epochs 300 \
# --batch-size 256 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee log_test.txt

# python main.py --no_distributed --no_distill --hf_model \
# --model google/vit-base-patch16-224-in21k  --teacher_model deit3_small --epochs 300 \
# --batch-size 1024 --data-path ../imagenet --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./results | tee log_mask50.txt

# python3 run_timm_example.py \
#     --model_name_or_path "timm/fastvit_t8.apple_in1k" \
#     --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" \
#     --warmup 3 \
#     --n_iterations 100 \
#     --use_hpu_graphs \
#     --bf16 \
#     --print_result | tee run_timm_example.txt

# python3 run_timm_example.py \
#     --model_name_or_path "deit_tiny_patch16_224" \
#     --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" \
#     --warmup 3 \
#     --n_iterations 100 \
#     --use_hpu_graphs \
#     --bf16 \
#     --print_result | tee -a run_timm_example.txt

# python3 run_timm_example.py \
#     --model_name_or_path "deit_tiny_patch16_224" \
#     --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" \
#     --warmup 3 \
#     --n_iterations 100 \
#     --use_hpu_graphs \
#     --bf16 \
#     --student_model \
#     --print_result | tee -a run_timm_example.txt


# python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
# --model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
# --batch-size 128 --data-path ./dataset --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
# --output_dir ./

