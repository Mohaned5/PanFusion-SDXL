from .PanoGenerator import PanoGenerator
import torch
import os
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel


class PanoOnly(PanoGenerator):
    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        self.mv_base_model = MultiViewBaseModel(None, pano_unet, None, cn, self.hparams.unet_pad)

    def init_noise(self, bs, equi_h, equi_w, device):
        return torch.randn(bs, 1, 4, equi_h, equi_w, device=device)

    def embed_prompt(self, batch):
        pano_prompt = self.get_pano_prompt(batch)
        pano_prompt_embd_tuple = self.encode_text(pano_prompt)
        pano_prompt_embd_1 = pano_prompt_embd_tuple[0][:, None]
        pano_prompt_embd_2 = pano_prompt_embd_tuple[1][:, None]
        pano_pooled_prompt_embd = pano_prompt_embd_tuple[2][:, None]
        return pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        b = batch['pano'].shape[0]
        h_pano, w_pano = batch['pano'].shape[-2:]

        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=pano_latent.device).long()

        pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd = self.embed_prompt(batch)

        original_size = (h_pano, w_pano)
        target_size = (h_pano, w_pano)
        crops_coords_top_left = (0, 0)
        add_time_ids = self._compute_time_ids(original_size, crops_coords_top_left, target_size, device=device)
        add_time_ids = add_time_ids.repeat(b, 1)

        pano_noise = self.init_noise(b, *pano_latent.shape[-2:], device=device)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)

        _, denoise = self.mv_base_model(
            pers_latents=None,
            pano_latent=pano_noise_z,
            timestep=t,
            pers_prompt_embd=None,
            pers_pooled_prompt_embd=None,
            pano_prompt_embd=pano_prompt_embd_1,
            pano_prompt_embd_2=pano_prompt_embd_2, 
            pano_pooled_prompt_embd=pano_pooled_prompt_embd,
            add_time_ids=add_time_ids, 
            cameras=None,
            pers_layout_cond=None,
            pano_layout_cond=batch.get('pano_layout_cond')
        )

        loss = torch.nn.functional.mse_loss(denoise, pano_noise)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss_pano', loss)
        return loss

    @torch.no_grad()
    def forward_cls_free(self, pano_latent, timestep, pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd, add_time_ids, layout_cond=None):
        pano_latent, timestep, layout_cond, pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd, add_time_ids = self.gen_cls_free_guide_pair(
            pano_latent, timestep, layout_cond, pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd, add_time_ids
        )

        _, pano_noise_pred = self.mv_base_model(
            pers_latents=None,
            pano_latent=pano_latent,
            timestep=timestep,
            pers_prompt_embd=None,
            pers_pooled_prompt_embd=None,
            pano_prompt_embd=pano_prompt_embd_1,
            pano_prompt_embd_2=pano_prompt_embd_2,
            pano_pooled_prompt_embd=pano_pooled_prompt_embd,
            add_time_ids=add_time_ids,
            cameras=None,
            pers_layout_cond=None,
            pano_layout_cond=layout_cond
        )

        pano_noise_pred = self.combine_cls_free_guide_pred(pano_noise_pred)
        return pano_noise_pred

    @torch.no_grad()
    def inference(self, batch):
        bs = batch['cameras']['height'].shape[0]
        h_pano, w_pano = batch['height'], batch['width']
        device = self.device

        pano_latent = self.init_noise(bs, h_pano//8, w_pano//8, device=device)

        pano_prompt_embd_1, pano_prompt_embd_2, pano_pooled_prompt_embd = self.embed_prompt(batch)
        prompt_null_1, prompt_null_2, pooled_null = self.encode_text('')
        prompt_null_1 = prompt_null_1[:, None]
        prompt_null_2 = prompt_null_2[:, None]
        pooled_null = pooled_null[:, None]

        pano_prompt_embd_1 = torch.cat([prompt_null_1.repeat(bs, 1, 1, 1), pano_prompt_embd_1])
        pano_prompt_embd_2 = torch.cat([prompt_null_2.repeat(bs, 1, 1, 1), pano_prompt_embd_2])
        pano_pooled_prompt_embd = torch.cat([pooled_null.repeat(bs, 1, 1), pano_pooled_prompt_embd])

        original_size = (h_pano, w_pano)
        target_size = (h_pano, w_pano)
        crops_coords_top_left = (0, 0)
        add_time_ids = self._compute_time_ids(original_size, crops_coords_top_left, target_size, device=device)

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            pano_latent = self.rotate_latent(pano_latent)
            curr_rot += self.hparams.rot_diff

            timestep_input = t[None].to(device)

            current_layout_cond = None
            if self.hparams.layout_cond:
                current_layout_cond = self.rotate_latent(pano_layout_cond)

            pano_noise_pred = self.forward_cls_free(
                pano_latent,
                timestep_input,
                pano_prompt_embd_1,
                pano_prompt_embd_2,
                pano_pooled_prompt_embd,
                add_time_ids,
                current_layout_cond
            )

            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent = self.rotate_latent(pano_latent, -curr_rot)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
        pano_pred = tensor_to_image(pano_pred)

        return pano_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pano_pred = self.inference(batch)
        self.log_val_image(pano_pred, batch['pano'], batch['pano_prompt'],
                           batch.get('pano_layout_cond'))

    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, pano_pred, pano, pano_prompt,
                      pano_layout_cond=None):
        log_dict = {
            'val/pano_pred': self.temp_wandb_image(
                pano_pred[0, 0], pano_prompt[0] if pano_prompt else None),
            'val/pano_gt': self.temp_wandb_image(
                pano[0, 0], pano_prompt[0] if pano_prompt else None),
        }
        if pano_layout_cond is not None:
            log_dict['val/pano_layout_cond'] = self.temp_wandb_image(
                pano_layout_cond[0, 0], pano_prompt[0] if pano_prompt else None)
        self.logger.experiment.log(log_dict)
