# PanFusion-SDXL

### Taming Stable Diffusion XL for Text to 360° Panorama Image Generation
Mohaned Abdulmahmood (SDXL migration), Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, Jianfei Cai

### [Project Page](https://chengzhag.github.io/publication/panfusion) | [Paper](https://arxiv.org/abs/2404.07949)

## Introduction

This repository contains the data preprocessing, training, testing, and evaluation code for our CVPR 2024 paper, adapted for Stable Diffusion XL (SDXL). PanFusion-SDXL leverages the enhanced capabilities of SDXL to generate high-quality 360° panorama images from text prompts, building on the original PanFusion framework.

## Installation

We use Anaconda to manage the environment. You can create the environment by running the following command:

```bash
git clone https://github.com/chengzhag/PanFusion
cd PanFusion
conda env create -f environment_sdxl.yaml
conda activate panfusion_sdxl
```

If you encounter issues with conda solving the environment or package versions, you can try creating the environment with specific package versions:

```bash
conda env create -f environment_sdxl_strict.yaml
```

We use [wandb](https://www.wandb.com/) to log and visualize the training process. Create an account and log in to wandb with:

```bash
wandb login
```

Refer to the wandb [report](https://wandb.ai/pidan1231239/pano_diffusion/reports/PanFusion--Vmlldzo3NzM1OTYy?accessToken=mmneovtrelnqd21gw5sk2cp8j0av65meohuf0ua850398sivq7duvkcvu934qlbt) for troubleshooting when reproducing results.

## Demo

The pretrained SDXL-based checkpoints will be available soon at [COMING SOON](#). Once available, place the checkpoint (`last.ckpt`) in the `logs/4142dlo4/checkpoints` folder. Then run the following command to test the model:

```bash
WANDB_MODE=offline WANDB_RUN_ID=4142dlo4 python main.py predict --data=Matterport3D --model=PanFusionSDXL --ckpt_path=last
```

Generated images will be saved in the `logs/4142dlo4/predict` folder.

For out-of-domain prompt testing:

```bash
WANDB_MODE=offline WANDB_RUN_ID=4142dlo4 python main.py predict --data=Demo --model=PanFusionSDXL --ckpt_path=last
```

## Data Preparation

### Download Data

We follow [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) to download the [Matterport3D](https://niessner.github.io/Matterport/) skybox dataset. Sign the form to request the download script `download_mp.py` and place it in the `data/Matterport3D` folder. Then run:

```bash
cd data/Matterport3D
python download_mp.py -o ./Matterport3D --type matterport_skybox_images
python unzip_skybox.py
```

Download the [splits](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0) provided by [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_skybox.tar
```

### Stitch Matterport3D Skybox

Stitch the Matterport3D skybox images into equirectangular projection images for training:

```bash
python -m scripts.stitch_mp3d
```

Stitched images are saved in `data/Matterport3D/mp3d_skybox/*/matterport_stitched_images`.

### Caption Images

Download perspective image captions from [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) [mp3d_skybox.tar](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_skybox.tar
```

For training, download BLIP-generated captions for equirectangular images [mp3d_stitched_caption.tar](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/Ec1A8tOmt_5ItvT2aktSUioBHzC_LRYjqaHPqipJuUhPHw?e=BgDGhL) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_stitched_caption.tar
```

<details>
<summary>Do it yourself</summary>

Generate captions yourself:

```bash
python -m scripts.caption_mp3d
```

</details>
<br>

### Render Layout

Download generated layout renderings [mp3d_layout.tar](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EQK5DP7LwWdOvhVjFER6dSsB255dUJknnVuNFROBEaWgjA?e=97UQEI) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_layout.tar
```

<details>
<summary>Do it yourself</summary>

Download and render layout labels:

```bash
cd data
git clone https://github.com/ericsujw/Matterport3DLayoutAnnotation
cd Matterport3DLayoutAnnotation
unzip label_data.zip
cd ../..
python -m scripts.render_layout
```

</details>
<br>

### Align Matterport3D Images

Download the Matlab alignment [tool](https://drive.google.com/file/d/1u6E5dT6zqFZsoLdV9ys-m0YJ9G3dtij7/view) to the `external` folder and unzip:

```bash
cd external
unzip preprocess.zip
```

Run the Matlab script `preprocess_mp3d.m` to align Matterport3D images.

## Training and Testing

### FAED

Download the pretrained FAED checkpoint [faed.ckpt](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EWMxyeTXtjlPnd7zmT36XqsBkmvLo_wxCmeKVAWIWTqUWg?e=Rtq1a4) to the `weights` folder for evaluating panorama image quality.

<details>
<summary>Do it yourself</summary>

Train FAED:

```bash
WANDB_NAME=faed python main.py fit --data=Matterport3D --model=FAED --trainer.max_epochs=60 --data.batch_size=4
```

Copy and rename the checkpoint to the `weights` folder. Training takes ~4 hours on a single NVIDIA A100 GPU.

</details>
<br>

### HorizonNet

Download the pretrained HorizonNet checkpoint [horizonnet.ckpt](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EYXsKsKuUqVLhfBgsnglKMIBgmVw9dvDVDUTH5l6wMZROg?e=gF1FW5) to the `weights` folder for layout-conditioned panorama evaluation.

<details>
<summary>Do it yourself</summary>

Download the official checkpoint [resnet50_rnn__st3d.pth](https://drive.google.com/file/d/1JcHwSlYVbrXW1oh37ze7sEvW9asPdI3A/view?usp=share_link) to `weights` and finetune:

```bash
WANDB_NAME=horizonnet python main.py fit --data=Matterport3D --model=HorizonNet --data.layout_cond_type=distance_map --data.horizon_layout=True --data.batch_size=4 --data.rand_rot_img=True --trainer.max_epochs=10 --model.ckpt_path=weights/resnet50_rnn__st3d.pth --data.num_workers=32
```

Copy and rename the checkpoint to `weights`. Training takes ~3 hours on a single NVIDIA A100 GPU.

</details>
<br>

### Text-to-Image Generation

Train the SDXL-based text-to-image generation model:

```bash
WANDB_NAME=panfusion_sdxl python main.py fit --data=Matterport3D --model=PanFusionSDXL
```

Training takes ~6 hours on 4x NVIDIA A100 GPUs. Logs are available at [wandb](https://wandb.ai/pidan1231239/pano_diffusion/runs/ad8103n1?nw=nwuserpidan1231239).

With `WANDB_RUN_ID` as `PANFUSION_SDXL_ID`, test the model:

```bash
WANDB_RUN_ID=<PANFUSION_SDXL_ID> python main.py test --data=Matterport3D --model=PanFusionSDXL --ckpt_path=last
WANDB_RUN_ID=<PANFUSION_SDXL_ID> python main.py test --data=Matterport3D --model=EvalPanoGen
```

Results are saved in `logs/<PANFUSION_SDXL_ID>/test` and logged to wandb.

### Layout-conditioned Panorama Generation

Finetune a ControlNet model for layout-conditioned panorama generation:

```bash
WANDB_NAME=panfusion_sdxl_lo python main.py fit --data=Matterport3D --model=PanFusionSDXL --trainer.max_epochs=100 --trainer.check_val_every_n_epoch=10 --model.ckpt_path=logs/<PANFUSION_SDXL_ID>/checkpoints/last.ckpt --model.layout_cond=True --data.layout_cond_type=distance_map --data.uncond_ratio=0.5
```

With `WANDB_RUN_ID` as `PANFUSION_SDXL_LO_ID`, test the model:

```bash
WANDB_RUN_ID=<PANFUSION_SDXL_LO_ID> python main.py test --data=Matterport3D --model=PanFusionSDXL --ckpt_path=last --model.layout_cond=True --data.layout_cond_type=distance_map
WANDB_RUN_ID=<PANFUSION_SDXL_LO_ID> python main.py test --data=Matterport3D --model=EvalPanoGen --data.manhattan_layout=True
```

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{panfusion2024,
  title={Taming Stable Diffusion for Text to 360◦ Panorama Image Generation},
  author={Zhang, Cheng and Wu, Qianyi and Cruz Gambardella, Camilo and Huang, Xiaoshui and Phung, Dinh and Ouyang, Wanli and Cai, Jianfei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```