python main.py \
--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
--batch-size 512 --data-path ../dataset/tiny-imagenet-200 --distillation-type soft \
--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
--output_dir ./


#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
#--model deit_tiny_patch16_224  --teacher_model deit3_small --epochs 300 \
#--batch-size 512 --data-path ./dataset --distillation-type soft \
#--distillation-alpha 0.5 --distillation-tau 1  --input-size 224 --len_num_keep 98 \
#--output_dir ./

