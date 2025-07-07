# [Domain-Adaptive Diagnosis of Lewy Body Disease with Transferability Aware Transformer](https://arxiv.org/pdf/2411.07794), MICCAI 2025

### updates (07/07/2025)
<!--  Add the environment requirements to reproduce the results.  --> 

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/MICCAI2025-TAT/blob/main/TAT-Method.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install timm==0.6.13
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)
- ViT-S_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-S_16.npz;tab=live_object?inv=1&invt=Ab2J2Q)


<!-- 
TVT with ViT-B_16 (ImageNet-21K) performs a little bit better than TVT with ViT-B_16 (ImageNet):
<p align="left"> 
<img width="500" src="https://github.com/uta-smile/TVT/blob/main/ImageNet_vs_ImageNet21K.png">
</p>
 --> 

### Datasets:

-The structural connectivity (SC) data for the AD dataset is provided in ADNI_SC.zip. Please unzip this file and update the corresponding paths in the files located in data/ADLBD_2to3.

-For the LBD dataset, SC was constructed at both 1-voxel and 2-voxel resolutions. The resulting data is available in LBD_SC_1voxel.zip and LBD_SC_2voxel.zip. Please unzip these files as well and update the paths accordingly in the files under data/ADLBD_2to3.

### Training:

All commands can be found in `script_2to3.txt`. An example:
```
python3 main.py --train_batch_size 16 --eval_batch_size 16 --dataset AD-LBD --name al --source_list data/ADLBD_2to3/adni_list_sc_noRemoveZeros_2to3.txt --target_list data/ADLBD_2to3/lbd_list_sc_noRemoveZeros_2to3.txt --test_list data/ADLBD_2to3/lbd_list_sc_noRemoveZeros_2to3.txt --s_num_classes 2 --t_num_classes 3 --model_type ViT-S_16 --pretrained_dir checkpoint/sam_ViT-S_16.npz --num_steps 10000 --beta 1.0 --gamma 0.01 --entropy_th 0.9 --learning_rate 0.07 --gpu_id 1 --use_cp --optimal 0 --perturbationRatio 0.0 --warmup_steps 1000
```

<!-- 
### Attention Map Visualization:
```
python3 visualize.py --dataset office --name wa --num_classes 31 --image_path att_visual.txt --img_size 256
```
The code will automatically use the best model in `wa` to visualize the attention maps of images in `att_visual.txt`. `att_visual.txt` contains image paths you want to visualize, for example:
```
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0001.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0002.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0003.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0004.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0005.jpg 5
```
 --> 

### Citation:
```
@article{yu2024FFTAT,
  title={Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation},
  author={Yu, Xiaowei and Huang, Zhe and Zhang, Zao},
  journal={arXiv preprint arXiv:2411.07794},
  year={2024}
}

@article{yu2023RCCT,
  title={Robust core-periphery constrained transformer for domain adaptation},
  author={Yu, Xiaowei and Zhu, Dajiang and Liu, Tianming},
  journal={arXiv preprint arXiv:2308.13515},
  year={2023}
}
```
Our code is largely borrowed from [FFTAT](https://github.com/Shawey94/WACV2025-FFTAT) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
