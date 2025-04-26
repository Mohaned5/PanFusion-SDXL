# --- START OF FILE Panime.py ---

import os
import json
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import cv2
import lightning as L
from torch.utils.data import DataLoader, Dataset, Subset
from einops import rearrange
from abc import abstractmethod
from utils.pano import Equirectangular, get_K_R
from .PanoDataset import PanoDataset, PanoDataModule

THETA_SEPARATION_DEG = 72.0
NUM_PERSPECTIVE_VIEWS = 20

class PanimeDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_dir = config['data_dir']
        self.result_dir = config.get('result_dir', None)
        self.config = config

        self.base_data = self.load_split(mode)
        self.data_len = len(self.base_data)

        if mode == 'predict':
             if config.get('repeat_predict', 1) > 1:
                 repeated_data = []
                 for i, d in enumerate(self.base_data):
                     for r in range(config['repeat_predict']):
                         new_d = d.copy()
                         new_d['repeat_id'] = r
                         repeated_data.append(new_d)
                 self.base_data = repeated_data
                 self.data_len = len(self.base_data)

        if not config.get('gt_as_result', False) and self.result_dir is not None and mode != 'predict':
            available_ids = self.scan_results(self.result_dir)
            original_ids = {d['pano_id'] for d in self.base_data}
            valid_ids = original_ids & available_ids
            if len(valid_ids) != len(original_ids):
                 print(f"WARNING: {len(original_ids)-len(valid_ids)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                 self.base_data = [d for d in self.base_data if d['pano_id'] in valid_ids]
                 self.data_len = len(self.base_data)

    def load_split(self, mode):
        if mode == 'predict':
            predict_file = os.path.join(self.data_dir, "predict.json")
            if not os.path.exists(predict_file):
                raise FileNotFoundError(f"Cannot find predict.json at {predict_file}")

            with open(predict_file, 'r') as f:
                all_data = json.load(f)

            new_data = []
            for sample in all_data:
                scene_id = sample["scene_id"]
                view_id = sample["view_id"]
                pano_prompt = sample.get("pano_prompt", "")
                new_data.append({
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "pano_prompt": pano_prompt
                })
            return new_data

        else:
            split_file = os.path.join(self.data_dir, f"{mode}.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Cannot find JSON split file: {split_file}")

            with open(split_file, 'r') as f:
                all_data = json.load(f)

            new_data = []
            for sample in all_data:
                if "pano" not in sample:
                     print(f"Skipping entry due to missing 'pano' key: {sample.get('pano_prompt', 'Unknown')}")
                     continue

                pano_filename = os.path.basename(sample["pano"])
                pano_id = os.path.splitext(pano_filename)[0]
                pano_path = os.path.join(self.data_dir, sample["pano"])

                images_paths = []
                if "images" in sample and isinstance(sample["images"], list):
                     images_paths = [os.path.join(self.data_dir, img) for img in sample["images"]]
                else:
                     print(f"Warning: Missing or invalid 'images' key for entry {pano_id}.")

                if not os.path.exists(pano_path):
                    print(f"Skipping entry {pano_id}: pano file missing at {pano_path}")
                    continue

                if "cameras" not in sample or not isinstance(sample["cameras"], dict) or \
                   'FoV' not in sample["cameras"] or 'theta' not in sample["cameras"] or 'phi' not in sample["cameras"] or \
                   not isinstance(sample["cameras"]['FoV'], list) or not sample["cameras"]['FoV'] or \
                   not isinstance(sample["cameras"]['theta'], list) or not sample["cameras"]['theta'] or \
                   not isinstance(sample["cameras"]['phi'], list) or not sample["cameras"]['phi']:
                     print(f"Skipping entry {pano_id}: Missing or invalid 'cameras' data.")
                     continue

                num_views_in_data = len(sample['cameras']['FoV'][0])
                if num_views_in_data != NUM_PERSPECTIVE_VIEWS:
                     print(f"Warning: Camera data for {pano_id} has {num_views_in_data} views, expected {NUM_PERSPECTIVE_VIEWS}. Skipping.")
                     continue

                if "prompts" not in sample or not isinstance(sample["prompts"], list):
                    print(f"Warning: Missing or invalid 'prompts' key for entry {pano_id}. Using empty prompts.")
                    prompts_list = [""] * NUM_PERSPECTIVE_VIEWS
                else:
                    prompts_list = sample["prompts"]
                    if len(prompts_list) != NUM_PERSPECTIVE_VIEWS:
                         print(f"Warning: Mismatch in number of prompts ({len(prompts_list)}) and expected views ({NUM_PERSPECTIVE_VIEWS}) for {pano_id}. Adjusting prompts.")
                         if len(prompts_list) > NUM_PERSPECTIVE_VIEWS:
                             prompts_list = prompts_list[:NUM_PERSPECTIVE_VIEWS]
                         else:
                             prompts_list.extend([""] * (NUM_PERSPECTIVE_VIEWS - len(prompts_list)))

                entry = {
                    "pano_id": pano_id,
                    "pano_path": pano_path,
                    "pano_prompt": sample.get("pano_prompt", ""),
                    "prompts": prompts_list,
                    "cameras_data": sample["cameras"]
                }
                new_data.append(entry)
            return new_data

    def scan_results(self, result_dir):
        folder_paths = glob(os.path.join(result_dir, '*'))
        results_ids = {
            os.path.basename(p)
            for p in folder_paths
            if os.path.isdir(p) and os.path.exists(os.path.join(p, 'pano.png'))
        }
        return results_ids


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.mode == 'predict':
            data = self.base_data[idx].copy()
            scene_id = data['scene_id']
            view_id = data['view_id']
            repeat_id_str = f"_{data['repeat_id']:06d}" if self.config.get('repeat_predict', 1) > 1 and 'repeat_id' in data else ""
            data['pano_id'] = f"{scene_id}_{view_id}{repeat_id_str}"
            data['pano_prompt'] = data.get('pano_prompt', "")
            data['height'] = self.config['pano_height']
            data['width'] = self.config['pano_height'] * 2
            return data

        base_data = self.base_data[idx].copy()
        pano_id_base = base_data['pano_id']

        try:
            equirectangular = Equirectangular.from_file(base_data['pano_path'])
            if equirectangular.equirectangular.ndim == 2:
                 equirectangular.equirectangular = cv2.cvtColor(equirectangular.equirectangular, cv2.COLOR_GRAY2RGB)
            elif equirectangular.equirectangular.shape[2] == 4:
                 equirectangular.equirectangular = cv2.cvtColor(equirectangular.equirectangular, cv2.COLOR_RGBA2RGB)
            elif equirectangular.equirectangular.shape[2] != 3:
                 raise ValueError(f"Unexpected number of channels: {equirectangular.equirectangular.shape[2]} in {base_data['pano_path']}")
        except Exception as e:
            print(f"Error loading panorama {base_data['pano_path']}: {e}")
            return {"error": f"Failed to load {pano_id_base}"}

        orig_fov = np.array(base_data['cameras_data']['FoV'][0], dtype=np.float32)
        orig_theta = np.array(base_data['cameras_data']['theta'][0], dtype=np.float32)
        orig_phi = np.array(base_data['cameras_data']['phi'][0], dtype=np.float32)
        orig_prompts = list(base_data['prompts'])

        do_flip = False
        roll_multiple = 0
        roll_degrees = 0.0

        if self.mode == 'train':
            do_flip = random.random() < 0.5
            try:
                 num_steps = int(round(360.0 / THETA_SEPARATION_DEG))
                 if num_steps <= 0:
                      num_steps = 1
            except ZeroDivisionError:
                 num_steps = 1

            roll_multiple = random.randint(0, num_steps - 1)
            roll_degrees = roll_multiple * THETA_SEPARATION_DEG

        new_theta = orig_theta.copy()
        new_phi = orig_phi.copy()
        new_prompts = list(orig_prompts)

        if self.mode == 'train' and roll_degrees > 0:
            equirectangular.rotate(roll_degrees)
            new_theta = (new_theta + roll_degrees + 360.0) % 360.0
            if roll_multiple != 0 and len(new_prompts) > 0:
                new_prompts = np.roll(new_prompts, roll_multiple).tolist()

        if self.mode == 'train' and do_flip:
            equirectangular.flip(True)
            new_theta = (180.0 - new_theta + 360.0) % 360.0

        processed_data = {
            'pano_id': pano_id_base,
            'pano_prompt': base_data['pano_prompt'],
            'height': self.config['pano_height'],
            'width': self.config['pano_height'] * 2,
        }

        new_Ks, new_Rs, perspective_images = [], [], []
        pers_h = self.config['pers_resolution']
        pers_w = self.config['pers_resolution']

        for i in range(len(new_theta)):
            K, R = get_K_R(orig_fov[i], new_theta[i], new_phi[i], pers_h, pers_w)
            new_Ks.append(K)
            new_Rs.append(R)
            try:
                pers_img = equirectangular.to_perspective(
                    (orig_fov[i], orig_fov[i]), new_theta[i], new_phi[i],
                    (pers_h, pers_w), mode='bilinear'
                )
                if pers_img.ndim == 2: pers_img = cv2.cvtColor(pers_img, cv2.COLOR_GRAY2RGB)
                elif pers_img.shape[2] == 4: pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGBA2RGB)
                perspective_images.append(pers_img)
            except Exception as e:
                 print(f"Error generating perspective view {i} for {processed_data['pano_id']} (theta={new_theta[i]}, phi={new_phi[i]}): {e}")
                 perspective_images.append(np.zeros((pers_h, pers_w, 3), dtype=np.uint8))

        if not perspective_images:
             print(f"Error: No perspective images generated for {processed_data['pano_id']}")
             return {"error": f"Failed perspective generation for {processed_data['pano_id']}"}

        pers_images_stack = np.stack(perspective_images)
        K_stack = np.stack(new_Ks)
        R_stack = np.stack(new_Rs)

        processed_data['cameras'] = {
            'height': np.full_like(new_theta, pers_h, dtype=int),
            'width': np.full_like(new_theta, pers_w, dtype=int),
            'FoV': orig_fov,
            'theta': new_theta,
            'phi': new_phi,
            'K': K_stack,
            'R': R_stack,
        }
        processed_data['prompts'] = new_prompts

        pano_resized = cv2.resize(equirectangular.equirectangular, (processed_data['width'], processed_data['height']), interpolation=cv2.INTER_AREA)
        pano_normalized = (pano_resized.astype(np.float32) / 127.5) - 1.0
        pers_normalized = (pers_images_stack.astype(np.float32) / 127.5) - 1.0

        processed_data['pano'] = rearrange(pano_normalized, 'h w c -> 1 c h w')
        processed_data['images'] = rearrange(pers_normalized, 'b h w c -> b c h w')

        if self.mode == 'train' and self.result_dir is None and random.random() < self.config.get('uncond_ratio', 0.0):
            processed_data['pano_prompt'] = ""
            processed_data['prompts'] = [""] * len(new_prompts)

        if self.config.get('gt_as_result', False):
             processed_data['pano_pred'] = processed_data['pano'].copy()
             processed_data['images_pred'] = processed_data['images'].copy()
        elif self.result_dir is not None:
             pano_pred_path = os.path.join(self.result_dir, pano_id_base, 'pano.png')
             if os.path.exists(pano_pred_path):
                 try:
                     pred_equi = Equirectangular.from_file(pano_pred_path)
                     if pred_equi.equirectangular.ndim == 2: pred_equi.equirectangular = cv2.cvtColor(pred_equi.equirectangular, cv2.COLOR_GRAY2RGB)
                     elif pred_equi.equirectangular.shape[2] == 4: pred_equi.equirectangular = cv2.cvtColor(pred_equi.equirectangular, cv2.COLOR_RGBA2RGB)

                     pred_pano_resized = cv2.resize(pred_equi.equirectangular, (processed_data['width'], processed_data['height']))
                     processed_data['pano_pred'] = rearrange(pred_pano_resized, 'h w c -> 1 c h w')

                     images_pred = []
                     for i in range(NUM_PERSPECTIVE_VIEWS):
                         img_pred_path = os.path.join(self.result_dir, pano_id_base, f"{i}.png")
                         if os.path.exists(img_pred_path):
                              img_pred = np.array(Image.open(img_pred_path).convert('RGB'))
                              img_pred_resized = cv2.resize(img_pred, (pers_w, pers_h))
                              images_pred.append(img_pred_resized)
                         else:
                              images_pred = []
                              break
                     if images_pred:
                          processed_data['images_pred'] = rearrange(np.stack(images_pred), 'b h w c -> b c h w')

                 except Exception as e:
                     print(f"Error loading prediction for {pano_id_base}: {e}")
                     processed_data.pop('pano_pred', None)
                     processed_data.pop('images_pred', None)

        if 'pano' in processed_data and isinstance(processed_data['pano'], np.ndarray):
            processed_data['pano'] = torch.from_numpy(processed_data['pano']).float()
        if 'images' in processed_data and isinstance(processed_data['images'], np.ndarray):
            processed_data['images'] = torch.from_numpy(processed_data['images']).float()
        if 'pano_pred' in processed_data and isinstance(processed_data['pano_pred'], np.ndarray):
             processed_data['pano_pred'] = torch.from_numpy(processed_data['pano_pred']).float()
        if 'images_pred' in processed_data and isinstance(processed_data['images_pred'], np.ndarray):
             processed_data['images_pred'] = torch.from_numpy(processed_data['images_pred']).float()

        if 'cameras' in processed_data:
            cameras = processed_data['cameras']
            if 'K' in cameras and isinstance(cameras['K'], np.ndarray):
                cameras['K'] = torch.from_numpy(cameras['K']).float()
            if 'R' in cameras and isinstance(cameras['R'], np.ndarray):
                cameras['R'] = torch.from_numpy(cameras['R']).float()

        return processed_data

class PanimeDataModule(PanoDataModule):
    """
    Data module using the augmented PanimeDataset.
    Loads train/val/test/predict splits from JSON files.
    Applies random augmentations during training.
    """
    def __init__(
        self,
        data_dir: str = 'data/Panime',
        fov: int = 90,
        pers_resolution: int = 512,
        pano_height: int = 1024,
        uncond_ratio: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 8,
        result_dir: str = None,
        gt_as_result: bool = False,
        repeat_predict: int = 1,
        **kwargs
    ):
        super(PanoDataModule, self).__init__()
        self.save_hyperparameters(
            'data_dir', 'fov', 'pers_resolution', 'pano_height',
            'uncond_ratio', 'batch_size', 'num_workers', 'result_dir',
            'gt_as_result', 'repeat_predict'
        )
        self.dataset_cls = PanimeDataset
        # Create hparams dict compatible with PanimeDataset
        self.dataset_hparams = {k: self.hparams.get(k) for k in [
            'data_dir', 'pers_resolution', 'pano_height',
            'uncond_ratio', 'result_dir', 'gt_as_result', 'repeat_predict'
            # Add 'fov' if needed by dataset directly
        ]}

    def setup(self, stage=None):
        # Setup datasets for different stages
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.dataset_hparams, mode='train')
            print(f"Train dataset size: {len(self.train_dataset)}")

        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(self.dataset_hparams, mode='val')
            print(f"Validation dataset size: {len(self.val_dataset)}") # Will show base size

        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(self.dataset_hparams, mode='test')
            print(f"Test dataset size: {len(self.test_dataset)}") # Will show base size

        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.dataset_hparams, mode='predict')
            print(f"Predict dataset size: {len(self.predict_dataset)}")

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'): self.setup('fit')
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'): self.setup('validate')
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'): self.setup('test')
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True
        )

    def predict_dataloader(self):
        if not hasattr(self, 'predict_dataset'): self.setup('predict')
        return DataLoader(
            self.predict_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False
        )