CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--model deit_small_patch16_224  --teacher_model deit_base --epochs 300 \
--batch-size 256 --data-path /data/ILSVRC2012 --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --maskedkd --len_num_keep 98 \
--output_dir ./output_test | tee ./test_log.txt