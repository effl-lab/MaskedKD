# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
# --model deit_small_patch16_224  --teacher_model deit_base --epochs 300 \
# --batch-size 256 --data-path /data/ILSVRC2012 --distillation-type soft \
# --distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --maskedkd --len_num_keep 98 \
# --output_dir ./output_test | tee ./test_log.txt

# mpirun -n 4 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u main.py \
--model deit_tiny_patch16_224 --teacher_model deit3_small --epochs 2 \
--batch-size 256 --data-path ../imagenet --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1 --input-size 224 --maskedkd --len_num_keep 98 \
--output_dir ./results | tee -a log_.txt