python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 256 --data-path ../dataset/imagenet --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./results | tee log_mask50.txt

python main.py --no_distributed \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 256 --data-path ../dataset/imagenet --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./results | tee log_mask50.txt



python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 128 --data-path ./dataset --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./

