from .PanoGenerator import PanoGenerator
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
import torch
import os
from PIL import Image
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
import torch.nn.functional as F

class PanFusion(PanoGenerator):
    def __init__(
            self,
            use_pers_prompt: bool = True,
            use_pano_prompt: bool = True,
            copy_pano_prompt: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        unet, pers_cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(unet, pano_unet, pers_cn, cn, self.hparams.unet_pad)
        if not self.hparams.layout_cond:
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def init_noise(self, bs, equi_h, equi_w, pers_h, pers_w, cameras, device):
        if 'FoV' not in cameras or cameras['FoV'].ndim < 2:
            raise ValueError("Camera FoV is missing or has unexpected dimensions.")
        num_views_per_item = cameras['FoV'].shape[1]

        cameras_flat = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        total_views = cameras_flat['FoV'].shape[0]

        pano_noise = torch.randn(bs, 1, 4, equi_h, equi_w, device=device)

        pano_noises_expanded = pano_noise.expand(-1, num_views_per_item, -1, -1, -1)
        pano_noises_flat = rearrange(pano_noises_expanded, 'b m c h w -> (b m) c h w')

        noise_flat = e2p(
            pano_noises_flat,
            cameras_flat['FoV'],
            cameras_flat['theta'],
            cameras_flat['phi'],
            (pers_h, pers_w),
            mode='nearest'
        )

        try:
            noise = rearrange(noise_flat, '(b m) c h w -> b m c h w', b=bs, m=num_views_per_item)
        except Exception as e:
            print(f"!!! Error during final rearrange in init_noise !!!")
            print(f"Input shape: {noise_flat.shape}, Target b={bs}, Target m={num_views_per_item}")
            raise e

        return pano_noise, noise

    def embed_prompt(self, batch, num_cameras):
        pers_pooled_prompt_embd = None
        pers_prompt_embd_1 = None
        pers_prompt_embd_2 = None
        if self.hparams.use_pers_prompt:
            pers_prompt = self.get_pers_prompt(batch)
            pers_embd_tuple = self.encode_text(pers_prompt)
            pers_prompt_embd_1 = rearrange(pers_embd_tuple[0], '(b m) l c -> b m l c', m=num_cameras)
            pers_prompt_embd_2 = rearrange(pers_embd_tuple[1], '(b m) l c -> b m l c', m=num_cameras)
            pers_pooled_prompt_embd = rearrange(pers_embd_tuple[2], '(b m) c -> b m c', m=num_cameras)
        else:
            pers_prompt = ''
            pers_embd_tuple = self.encode_text(pers_prompt)
            pers_prompt_embd_1 = pers_embd_tuple[0][:, None].repeat(1, num_cameras, 1, 1)
            pers_prompt_embd_2 = pers_embd_tuple[1][:, None].repeat(1, num_cameras, 1, 1)
            pers_pooled_prompt_embd = pers_embd_tuple[2][:, None].repeat(1, num_cameras, 1)

        pano_pooled_prompt_embd = None
        pano_prompt_embd_1 = None
        pano_prompt_embd_2 = None
        if self.hparams.use_pano_prompt:
            pano_prompt = self.get_pano_prompt(batch)
        else:
            pano_prompt = ''

        pano_embd_tuple = self.encode_text(pano_prompt)
        pano_prompt_embd_1 = pano_embd_tuple[0][:, None] 
        pano_prompt_embd_2 = pano_embd_tuple[1][:, None]
        pano_pooled_prompt_embd = pano_embd_tuple[2][:, None]

        return (pers_prompt_embd_1, pers_prompt_embd_2, pers_pooled_prompt_embd), \
               (pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd)

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        torch.autograd.set_detect_anomaly(True)

        latents = self.encode_image(batch['images'], self.vae)
        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)

        b, m, c, h_lat, w_lat = latents.shape
        _, _, _, h_pano_lat, w_pano_lat = pano_latent.shape
        h_img, w_img = batch['images'].shape[-2:]
        h_pano_img, w_pano_img = batch['pano'].shape[-2:]

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=latents.device).long()

        (pers_prompt_embd_1, pers_prompt_embd_2, pers_pooled_prompt_embd), \
        (pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd) = self.embed_prompt(batch, m)

        original_size = (h_img, w_img)
        target_size = (h_img, w_img)
        crops_coords_top_left = (0, 0)
        add_time_ids = self._compute_time_ids(original_size, crops_coords_top_left, target_size, device=device)
        add_time_ids = add_time_ids.repeat(b * m, 1)

        pano_original_size = (h_pano_img, w_pano_img)
        pano_target_size = (h_pano_img, w_pano_img)
        pano_add_time_ids = self._compute_time_ids(pano_original_size, crops_coords_top_left, pano_target_size, device=device)
        pano_add_time_ids = pano_add_time_ids.repeat(b, 1)

        pano_noise, noise = self.init_noise(
            b, h_pano_lat, w_pano_lat, h_lat, w_lat, batch['cameras'], device)

        noise_z = self.scheduler.add_noise(latents, noise, t)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)

        t_pers = t[:, None].repeat(1, m)
        t_pano = t 

        pers_layout_cond=batch.get('images_layout_cond')
        pano_layout_cond=batch.get('pano_layout_cond')

        denoise, pano_denoise = self.mv_base_model(
            pers_latents=noise_z,
            pano_latent=pano_noise_z,
            timestep=(t_pers, t_pano),
            pers_prompt_embd=pers_prompt_embd_1,
            pers_prompt_embd_2=pers_prompt_embd_2,
            pers_pooled_prompt_embd=pers_pooled_prompt_embd,
            pano_prompt_embd=pano_prompt_embd_1,
            pano_prompt_embd_2=pano_prompt_embd_2,
            pano_pooled_prompt_embd=pano_pooled_prompt_embd,
            add_time_ids=(add_time_ids, pano_add_time_ids),
            cameras=batch['cameras'],
            pers_layout_cond=pers_layout_cond,
            pano_layout_cond=pano_layout_cond
         )

        loss_pers = F.mse_loss(denoise, noise)
        loss_pano = F.mse_loss(pano_denoise, pano_noise)
        loss = loss_pers + loss_pano

        self.log('train/loss', loss, prog_bar=False, sync_dist=True)
        self.log('train/loss_pers', loss_pers, prog_bar=True, sync_dist=True)
        self.log('train/loss_pano', loss_pano, prog_bar=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def forward_cls_free(self, latents, pano_latent, timestep,
                        pers_prompt_embd_1, pers_prompt_embd_2, pers_pooled_prompt_embd,
                        pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd,
                        add_time_ids, cameras,
                        images_layout_cond=None, pano_layout_cond=None):
        bs = latents.shape[0]
        m = latents.shape[1]

        add_time_ids_pers, add_time_ids_pano = add_time_ids

        prompt_null_1, prompt_null_2, pooled_null = self.encode_text('')
        prompt_null_1 = prompt_null_1[:, None]
        prompt_null_2 = prompt_null_2[:, None]
        pooled_null = pooled_null[:, None]

        if pers_prompt_embd_1 is not None:
            pers_prompt_null_1 = prompt_null_1.repeat(bs, m, 1, 1) 
            pers_prompt_null_2 = prompt_null_2.repeat(bs, m, 1, 1) 
            pers_pooled_null = pooled_null.repeat(bs, m, 1)  
            pers_prompt_embd_1_cfg = torch.cat([pers_prompt_null_1, pers_prompt_embd_1], dim=0) 
            pers_prompt_embd_2_cfg = torch.cat([pers_prompt_null_2, pers_prompt_embd_2], dim=0)
            pers_pooled_prompt_embd_cfg = torch.cat([pers_pooled_null, pers_pooled_prompt_embd], dim=0) 
        else:
            pers_prompt_embd_1_cfg = None
            pers_prompt_embd_2_cfg = None
            pers_pooled_prompt_embd_cfg = None

        if pano_prompt_embd_1 is not None:
            pano_prompt_null_1 = prompt_null_1.repeat(bs, 1, 1, 1)
            pano_prompt_null_2 = prompt_null_2.repeat(bs, 1, 1, 1)
            pano_pooled_null = pooled_null.repeat(bs, 1, 1)
            pano_prompt_embd_1_cfg = torch.cat([pano_prompt_null_1, pano_prompt_embd_1], dim=0)
            pano_prompt_embd_2_cfg = torch.cat([pano_prompt_null_2, pano_prompt_embd_2], dim=0) 
            pano_pooled_prompt_embd_cfg = torch.cat([pano_pooled_null, pano_pooled_prompt_embd], dim=0) 
        else:
            pano_prompt_embd_1_cfg = None
            pano_prompt_embd_2_cfg = None
            pano_pooled_prompt_embd_cfg = None

        latents_cfg = torch.cat([latents, latents], dim=0)
        pano_latent_cfg = torch.cat([pano_latent, pano_latent], dim=0)
        if timestep.numel() == 1:
            timestep_cfg = timestep.repeat(bs * 2) 
        elif timestep.numel() == bs:
            timestep_cfg = torch.cat([timestep, timestep], dim=0)
        else:
            raise ValueError(f"Unexpected input timestep shape {timestep.shape} in forward_cls_free for bs={bs}")

        cameras_cfg = {k: torch.cat([v, v], dim=0) for k, v in cameras.items()} 

        images_layout_cond_cfg = torch.cat([images_layout_cond, images_layout_cond], dim=0) if images_layout_cond is not None else None
        pano_layout_cond_cfg = torch.cat([pano_layout_cond, pano_layout_cond], dim=0) if pano_layout_cond is not None else None

        add_time_ids_pers_cfg = torch.cat([add_time_ids_pers, add_time_ids_pers], dim=0)
        add_time_ids_pano_cfg = torch.cat([add_time_ids_pano, add_time_ids_pano], dim=0)
        add_time_ids_cfg = (add_time_ids_pers_cfg, add_time_ids_pano_cfg)

        noise_pred, pano_noise_pred = self.mv_base_model(
            pers_latents=latents_cfg,
            pano_latent=pano_latent_cfg,
            timestep=timestep_cfg,
            pers_prompt_embd=pers_prompt_embd_1_cfg,
            pers_prompt_embd_2=pers_prompt_embd_2_cfg,
            pers_pooled_prompt_embd=pers_pooled_prompt_embd_cfg,
            pano_prompt_embd=pano_prompt_embd_1_cfg,
            pano_prompt_embd_2=pano_prompt_embd_2_cfg,
            pano_pooled_prompt_embd=pano_pooled_prompt_embd_cfg,
            add_time_ids=add_time_ids_cfg,
            cameras=cameras_cfg,
            pers_layout_cond=images_layout_cond_cfg,
            pano_layout_cond=pano_layout_cond_cfg
        )

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        pano_noise_pred_uncond, pano_noise_pred_text = pano_noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.hparams.guidance_scale * (noise_pred_text - noise_pred_uncond)
        pano_noise_pred = pano_noise_pred_uncond + self.hparams.guidance_scale * (pano_noise_pred_text - pano_noise_pred_uncond)

        return noise_pred, pano_noise_pred

    def rotate_latent(self, pano_latent, cameras, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent, cameras

        pano_latent = super().rotate_latent(pano_latent, degree)
        cameras = cameras.copy()
        cameras['theta'] = (cameras['theta'] + degree) % 360
        return pano_latent, cameras

    @torch.no_grad()
    @rank_zero_only
    def log_val_image_np(self, images_pred_np, images_gt_np, pano_pred_np, pano_gt_np, pano_prompt,
                         images_layout_np=None, pano_layout_np=None):
        log_dict = {}

        def np_to_wandb_image(img_np, caption=None):
             if img_np is None: return None
             if img_np.ndim == 4 and img_np.shape[0] == 1:
                 img_np = img_np.squeeze(0)
             elif img_np.ndim == 5 and img_np.shape[0] == 1 and img_np.shape[1] == 1: 
                  img_np = img_np.squeeze(0).squeeze(0)

             if img_np.ndim != 3:
                  print(f"Warning: Unexpected numpy array ndim {img_np.ndim} for wandb logging.")
                  return None 
             if img_np.shape[-1] != 3:
                  print(f"Warning: Unexpected channel count {img_np.shape[-1]} for wandb logging.")
                  return None 

             try:
                 im = Image.fromarray(img_np)
                 return self.temp_wandb_image(img_np, caption)
             except Exception as e:
                 print(f"Error creating wandb.Image from numpy: {e}")
                 return None


        log_dict[f"val/pred"] = {
             "pers": [np_to_wandb_image(images_pred_np[0, i]) for i in range(images_pred_np.shape[1])],
             "pano": np_to_wandb_image(pano_pred_np[0, 0])
        }
        log_dict[f"val/gt"] = {
             "pers": [np_to_wandb_image(images_gt_np[0, i], caption=pano_prompt[0]) for i in range(images_gt_np.shape[1])],
             "pano": np_to_wandb_image(pano_gt_np[0, 0], caption=pano_prompt[0])
        }
        if images_layout_np is not None and pano_layout_np is not None:
             log_dict[f"val/layout_cond"] = {
                 "pers": [np_to_wandb_image(images_layout_np[0, i]) for i in range(images_layout_np.shape[1])],
                 "pano": np_to_wandb_image(pano_layout_np[0, 0])
             }

        flat_log_dict = {}
        if log_dict.get("val/pred"):
             flat_log_dict["val/pano_pred"] = log_dict["val/pred"]["pano"]
             # flat_log_dict["val/pers_pred"] = log_dict["val/pred"]["pers"] # Log list of images
        if log_dict.get("val/gt"):
             flat_log_dict["val/pano_gt"] = log_dict["val/gt"]["pano"]
             # flat_log_dict["val/pers_gt"] = log_dict["val/gt"]["pers"]
        if log_dict.get("val/layout_cond"):
             flat_log_dict["val/pano_layout_cond"] = log_dict["val/layout_cond"]["pano"]
             # flat_log_dict["val/pers_layout_cond"] = log_dict["val/layout_cond"]["pers"]

        final_log_dict = {k: v for k, v in flat_log_dict.items() if v is not None}
        if final_log_dict:
             self.logger.experiment.log(final_log_dict)
        else:
             print("Warning: No valid images to log for validation step.")


    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h_img, w_img = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()

        pano_h_val = batch['height']
        pano_w_val = batch['width']

        if isinstance(pano_h_val, torch.Tensor):
            if pano_h_val.numel() == 1:
                h_pano_img = pano_h_val.item()
            elif pano_h_val.numel() > 1:
                print(f"Warning: batch['height'] has multiple elements ({pano_h_val.numel()}). Using the first value: {pano_h_val[0].item()}")
                h_pano_img = pano_h_val[0].item()
            else:
                raise ValueError("batch['height'] tensor is empty!")
        else:
            h_pano_img = pano_h_val

        if isinstance(pano_w_val, torch.Tensor):
            if pano_w_val.numel() == 1:
                w_pano_img = pano_w_val.item() 
            elif pano_w_val.numel() > 1:
                print(f"Warning: batch['width'] has multiple elements ({pano_w_val.numel()}). Using the first value: {pano_w_val[0].item()}")
                w_pano_img = pano_w_val[0].item()
            else:
                raise ValueError("batch['width'] tensor is empty!")
        else:
            w_pano_img = pano_w_val 

        device = self.device

        h_lat, w_lat = h_img // 8, w_img // 8
        h_pano_lat, w_pano_lat = h_pano_img // 8, w_pano_img // 8

        pano_latent, latents = self.init_noise(
            bs, h_pano_lat, w_pano_lat, h_lat, w_lat, batch['cameras'], device)

        (pers_prompt_embd_1, pers_prompt_embd_2, pers_pooled_prompt_embd), \
        (pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd) = self.embed_prompt(batch, m)

        original_size = (h_img, w_img)
        target_size = (h_img, w_img)
        crops_coords_top_left = (0, 0)
        add_time_ids_pers = self._compute_time_ids(original_size, crops_coords_top_left, target_size, device=device)
        add_time_ids_pers = add_time_ids_pers.repeat(bs * m, 1)

        pano_original_size = (h_pano_img, w_pano_img)
        pano_target_size = (h_pano_img, w_pano_img)
        add_time_ids_pano = self._compute_time_ids(pano_original_size, crops_coords_top_left, pano_target_size, device=device)
        add_time_ids_pano = add_time_ids_pano.repeat(bs, 1)
        add_time_ids = (add_time_ids_pers, add_time_ids_pano)

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')
        images_layout_cond = batch.get('images_layout_cond')

        curr_rot = 0

        for i, t in enumerate(timesteps):
            timestep_input = t.to(device) 

            pano_latent_rotated, batch_cameras_rotated = self.rotate_latent(pano_latent, batch['cameras'])
            curr_rot = (curr_rot + self.hparams.rot_diff) % 360

            current_pano_layout_cond = None
            current_images_layout_cond = None

            if self.hparams.layout_cond:
                if pano_layout_cond is not None:
                    current_pano_layout_cond = super(PanFusion, self).rotate_latent(pano_layout_cond, degree=self.hparams.rot_diff)
                if self.hparams.pers_layout_cond and images_layout_cond is not None:
                    current_images_layout_cond = images_layout_cond
          
            noise_pred, pano_noise_pred = self.forward_cls_free(
                latents, pano_latent_rotated, timestep_input,
                pers_prompt_embd_1, pers_prompt_embd_2, pers_pooled_prompt_embd,
                pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd,
                add_time_ids, batch_cameras_rotated,
                current_images_layout_cond, current_pano_layout_cond
            )

            latents_prev = latents
            pano_latent_prev = pano_latent_rotated
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            pano_latent = self.scheduler.step(pano_noise_pred, t, pano_latent_rotated).prev_sample

        pano_latent, _ = self.rotate_latent(pano_latent, batch['cameras'], -curr_rot)

        images_pred_tensor = self.decode_latent(latents, self.vae)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad_tensor = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred_tensor = self.unpad_pano(pano_pred_pad_tensor)

        try:
            images_pred_np = tensor_to_image(images_pred_tensor)
        except Exception as e:
            raise ValueError("Failed converting perspective tensors to images") from e

        try:
            pano_pred_np = tensor_to_image(pano_pred_tensor)
        except Exception as e:
            raise ValueError("Failed converting panoramic tensor to image") from e

        return images_pred_np, pano_pred_np

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred_np, pano_pred_np = self.inference(batch)

        images_gt_np = tensor_to_image(batch['images'])
        pano_gt_np = tensor_to_image(batch['pano'])
        images_layout_np = tensor_to_image(batch.get('images_layout_cond')) if batch.get('images_layout_cond') is not None else None
        pano_layout_np = tensor_to_image(batch.get('pano_layout_cond')) if batch.get('pano_layout_cond') is not None else None

        self.log_val_image_np(
            images_pred_np, images_gt_np, pano_pred_np, pano_gt_np, batch['pano_prompt'],
            images_layout_np, pano_layout_np
        )


    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        _, pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, images, pano_pred, pano, pano_prompt,
                      images_layout_cond=None, pano_layout_cond=None):
        log_dict = {f"val/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt).items()}
        log_dict.update({f"val/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt).items()})
        if images_layout_cond is not None and pano_layout_cond is not None:
            log_dict.update({f"val/{k}_layout_cond": v for k, v in self.temp_wandb_images(
                images_layout_cond, pano_layout_cond, None, pano_prompt).items()})
        self.logger.experiment.log(log_dict)

    def temp_wandb_images(self, images, pano, prompt=None, pano_prompt=None):
        log_dict = {}
        pers = []
        for m_i in range(images.shape[1]):
            pers.append(self.temp_wandb_image(
                images[0, m_i], prompt[m_i][0] if prompt else None))
        log_dict['pers'] = pers

        log_dict['pano'] = self.temp_wandb_image(
            pano[0, 0], pano_prompt[0] if pano_prompt else None)
        return log_dict