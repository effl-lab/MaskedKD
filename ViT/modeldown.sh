#!/bin/bash

models=("ViT-B_16-224" "ViT-B_16" "ViT-B_32" "ViT-L_16-224" "ViT-L_16" "ViT-L_32")

base_url="https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012"

for model in "${models[@]}"; do
  echo "Downloading ${model}.npz..."
  wget "${base_url}/${model}.npz" -O "${model}.npz"
done

echo "All models downloaded."
