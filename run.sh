CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 512 --data-path /data/ILSVRC2012/ --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./