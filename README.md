# MaskedKD for Intel Gaudi (Masked Knowledge Distillation)

This is the implementation of a paper titled "The Role of Masking for Efficient Supervised Knowledge Distillation of Vision Transformers (ECCV 2024)" for Intel Gaudi.

### [Paper](https://arxiv.org/abs/2302.10494) | [Project page](https://maskedkd.github.io/)

<br>
Seungwoo Son, Jegwang Ryu, Namhoon Lee, Jaeho Lee <br>
Pohang University of Science and Technology (POSTECH)

## Summary
Our method, MaskedKD, reduces supervision cost by masking teacher ViT input based on student attention, maintaining student accuracy while saving computation.

<center>
<img src="./materials/maskedkd_main_figures.png"  style="zoom: 15%;"/>
</center>

This repo is based on [DeiT](https://github.com/facebookresearch/deit), [MAE](https://github.com/facebookresearch/mae) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
