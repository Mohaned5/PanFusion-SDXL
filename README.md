# PanFusion-SDXL

### Stable Diffusion XL for Text to 360° Panorama Image Generation
Migration by Mohaned Abdulmahmood  
Original Work by Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, Jianfei Cai

### [Project Page](https://chengzhag.github.io/publication/panfusion) | [Paper](https://arxiv.org/abs/2404.07949)

## Introduction

This repository is a public adaptation of the [PanFusion](https://github.com/chengzhag/PanFusion) project, originally presented in the CVPR 2024 paper, migrated from Stable Diffusion 2.0 (SD2.0) to Stable Diffusion XL (SDXL) by Mohaned Abdulmahmood. PanFusion-SDXL enables high-quality 360° panorama image generation from text prompts, leveraging SDXL's enhanced capabilities. It includes data preprocessing, training, testing, and evaluation code, adapted for SDXL.

## Installation

We use Anaconda to manage the environment. Clone the repository and create the environment:

```bash
git clone https://github.com/Mohaned5/PanFusion-SDXL
cd PanFusion-SDXL
conda env create -f environment_strict.yaml
conda activate panfusion
```

We use [wandb](https://www.wandb.com/) for logging and visualizing training. Create an account and log in:

```bash
wandb login
```

Refer to the original wandb [report](https://wandb.ai/pidan1231239/pano_diffusion/reports/PanFusion--Vmlldzo3NzM1OTYy?accessToken=mmneovtrelnqd21gw5sk2cp8j0av65meohuf0ua850398sivq7duvkcvu934qlbt) for troubleshooting.

## Demo

Pretrained SDXL-based checkpoints will be available at [COMING SOON](#). Once available, place the checkpoint (`last.ckpt`) in `logs/4142dlo4/checkpoints` and test the model:

```bash
WANDB_MODE=offline WANDB_RUN_ID=4142dlo4 python main.py predict --data=Matterport3D --model=PanFusion --ckpt_path=last
```

Generated images are saved in `logs/4142dlo4/predict`.

For out-of-domain prompts:

```bash
WANDB_MODE=offline WANDB_RUN_ID=4142dlo4 python main.py predict --data=Demo --model=PanFusion --ckpt_path=last
```
## Example Outputs

### Matterport3D checkpoint (incomplete - 5 epochs in)
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/420867e7-9005-455e-b09b-2fd31215e973" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/73ac5456-4f11-48e7-9175-4df55cfe3715" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/356e9555-c39c-47e4-97c9-94bfe171a2c9" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/7c839ad2-6504-4015-b24d-214a69f3e431" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/1508fd51-1cce-4de3-a0d6-a06d43fe74dd" />

### Panime fine-tuned checkpoint (complete)
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/0496d1bc-58fa-4695-b2ba-58252df27f2d" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/877b3151-3f1d-424d-b4ed-eddedb80f3b4" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/3f16405b-4dfb-46e0-8e01-22f72214cf74" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/b115ffe4-63ff-4d98-8f9e-47a5dd3f1373" />
<img width="2048" height="1024" alt="Image" src="https://github.com/user-attachments/assets/753c83f2-a19e-401d-83c6-4616a3c1a158" />

## Data Preparation

### Download Data

Follow [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) to download the [Matterport3D](https://niessner.github.io/Matterport/) skybox dataset. Request the `download_mp.py` script, place it in `data/Matterport3D`, and run:

```bash
cd data/Matterport3D
python download_mp.py -o ./Matterport3D --type matterport_skybox_images
python unzip_skybox.py
```

Download [splits](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0) from [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_skybox.tar
```

### Stitch Matterport3D Skybox

Stitch skybox images into equirectangular projections:

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

Download BLIP-generated captions for equirectangular images [mp3d_stitched_caption.tar](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/Ec1A8tOmt_5ItvT2aktSUioBHzC_LRYjqaHPqipJuUhPHw?e=BgDGhL) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_stitched_caption.tar
```

<details>
<summary>Do it yourself</summary>

Generate captions:

```bash
python -m scripts.caption_mp3d
```

</details>
<br>

### Render Layout

Download layout renderings [mp3d_layout.tar](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EQK5DP7LwWdOvhVjFER6dSsB255dUJknnVuNFROBEaWgjA?e=97UQEI) to `data/Matterport3D` and unzip:

```bash
cd data/Matterport3D
tar -xvf mp3d_layout.tar
```

<details>
<summary>Do it yourself</summary>

Download and render layouts:

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

Download the Matlab alignment [tool](https://drive.google.com/file/d/1u6E5dT6zqFZsoLdV9ys-m0YJ9G3dtij7/view) to `external` and unzip:

```bash
cd external
unzip preprocess.zip
```

Run `preprocess_mp3d.m` in Matlab to align images.

## Training and Testing

### FAED

Download the pretrained FAED checkpoint [faed.ckpt](https://monashuni-my.sharepoint.com/:copy and rename to `weights` folder for panorama quality evaluation.

<details>
<summary>Do it yourself</summary>

Train FAED:

```bash
WANDB_NAME=faed python main.py fit --data=Matterport3D --model=FAED --trainer.max_epochs=60 --data.batch_size=4
```

Copy and rename the checkpoint to `weights`. Training takes ~4 hours on an NVIDIA A100 GPU.

</details>
<br>

### HorizonNet

Download the pretrained HorizonNet checkpoint [horizonnet.ckpt](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EYXsKsKuUqVLhfBgsnglKMIBgmVw9dvDVDUTH5l6wMZROg?e=gF1FW5) to `weights` for layout-conditioned evaluation.

<details>
<summary>Do it yourself</summary>

Download [resnet50_rnn__st3d.pth](https://drive.google.com/file/d/1JcHwSlYVbrXW1oh37ze7sEvW9asPdI3A/view?usp=share_link) to `weights` and finetune:

```bash
WANDB_NAME=horizonnet python main.py fit --data=Matterport3D --model=HorizonNet --data.layout_cond_type=distance_map --data.horizon_layout=True --data.batch_size=4 --data.rand_rot_img=True --trainer.max_epochs=10 --model.ckpt_path=weights/resnet50_rnn__st3d.pth --data.num_workers=32
```

Copy and rename the checkpoint to `weights`. Training takes ~3 hours on an NVIDIA A100 GPU.

</details>
<br>

### Text-to-Image Generation

Train the SDXL-based model:

```bash
WANDB_NAME=panfusion python main.py fit --data=Matterport3D --model=PanFusion
```

Training takes ~6 hours on 4x NVIDIA A100 GPUs. Logs are available at [wandb](https://wandb.ai/pidan1231239/pano_diffusion/runs/ad8103n1?nw=nwuserpidan1231239).

Test with `WANDB_RUN_ID` as `PANFUSION_ID`:

```bash
WANDB_RUN_ID=<PANFUSION_ID> python main.py test --data=Matterport3D --model=PanFusion --ckpt_path=last
WANDB_RUN_ID=<PANFUSION_ID> python main.py test --data=Matterport3D --model=EvalPanoGen
```

Results are saved in `logs/<PANFUSION_ID>/test` and logged to wandb.

### Layout-conditioned Panorama Generation

Finetune a ControlNet model for layout-conditioned generation:

```bash
WANDB_NAME=panfusion python main.py fit --data=Matterport3D --model=PanFusion --trainer.max_epochs=100 --trainer.check_val_every_n_epoch=10 --model.ckpt_path=logs/<PANFUSION_ID>/checkpoints/last.ckpt --model.layout_cond=True --data.layout_cond_type=distance_map --data.uncond_ratio=0.5
```

Test with `WANDB_RUN_ID` as `PANFUSION_SDXL_LO_ID`:

```bash
WANDB_RUN_ID=<PANFUSION_SDXL_LO_ID> python main.py test --data=Matterport3D --model=PanFusion --ckpt_path=last --model.layout_cond=True --data.layout_cond_type=distance_map
WANDB_RUN_ID=<PANFUSION_SDXL_LO_ID> python main.py test --data=Matterport3D --model=EvalPanoGen --data.manhattan_layout=True
```

## Citation

If you find this work helpful, please cite the original PanFusion paper:

```bibtex
@inproceedings{panfusion2024,
  title={Taming Stable Diffusion for Text to 360◦ Panorama Image Generation},
  author={Zhang, Cheng and Wu, Qianyi and Cruz Gambardella, Camilo and Huang, Xiaoshui and Phung, Dinh and Ouyang, Wanli and Cai, Jianfei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgments

This repository is a migration of the [PanFusion](https://github.com/chengzhag/PanFusion) project to Stable Diffusion XL, made public by Mohaned Abdulmahmood. Thanks to the original authors for their foundational work.
