import os
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
from abc import abstractmethod
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
import copy
from utils.pano import pad_pano, unpad_pano
from ..modules.utils import WandbLightningModule
from diffusers import ControlNetModel
from diffusers.utils import USE_PEFT_BACKEND

if USE_PEFT_BACKEND:
    try:
        from peft import LoraConfig, inject_adapter_in_model
        print("PEFT backend detected and imported.")
    except ImportError:
        print("PEFT backend detected but import failed. Please ensure 'peft' is installed correctly.")
        USE_PEFT_BACKEND = False
else:
    print("Warning: PEFT backend not detected by diffusers. LoRA injection requires 'peft' library.")
    LoraConfig = None
    inject_adapter_in_model = None


class PanoBase(WandbLightningModule):
    def __init__(
            self,
            pano_prompt_prefix: str = '',
            pers_prompt_prefix: str = '',
            mv_pano_prompt: bool = False,
            copy_pano_prompt: bool = False,
            ):
        super().__init__()
        self.save_hyperparameters()

    def add_pano_prompt_prefix(self, pano_prompt):
        if isinstance(pano_prompt, str):
            if pano_prompt == '':
                return ''
            if self.hparams.pano_prompt_prefix == '':
                return pano_prompt
            return ' '.join([self.hparams.pano_prompt_prefix, pano_prompt])
        return [self.add_pano_prompt_prefix(p) for p in pano_prompt]

    def add_pers_prompt_prefix(self, pers_prompt):
        if isinstance(pers_prompt, str):
            if pers_prompt == '':
                return ''
            if self.hparams.pers_prompt_prefix == '':
                return pers_prompt
            return ' '.join([self.hparams.pers_prompt_prefix, pers_prompt])
        return [self.add_pers_prompt_prefix(p) for p in pers_prompt]

    def get_pano_prompt(self, batch):
        if self.hparams.mv_pano_prompt:
            prompts = list(map(list, zip(*batch['prompt'])))
            pano_prompt = ['. '.join(p1) if p2 else '' for p1, p2 in zip(prompts, batch['pano_prompt'])]
        else:
            pano_prompt = batch['pano_prompt']
        return self.add_pano_prompt_prefix(pano_prompt)

    def get_pers_prompt(self, batch):
        if self.hparams.copy_pano_prompt:
            prompts = sum([[p] * batch['cameras']['height'].shape[-1] for p in batch['pano_prompt']], [])
        else:
            prompts = sum(map(list, zip(*batch['prompt'])), [])
        return self.add_pers_prompt_prefix(prompts)


class PanoGenerator(PanoBase):
    def __init__(
            self,
            lr: float = 2e-5,
            guidance_scale: float = 7.5,
            model_id: str = 'stabilityai/stable-diffusion-xl-base-1.0',
            diff_timestep: int = 50,
            latent_pad: int = 16,
            pano_lora: bool = True,
            train_pano_lora: bool = True,
            pers_lora: bool = True,
            train_pers_lora: bool = True,
            lora_rank: int = 4,
            ckpt_path: str = None,
            rot_diff: float = 90.0,
            layout_cond: bool = False,
            pers_layout_cond: bool = False,
            unet_pad: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.trainable_params = []
        self.save_hyperparameters()
        self.load_shared()
        self.instantiate_model()
        if ckpt_path is not None:
            print(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path)['state_dict']
            self.convert_state_dict(state_dict)
            try:
                self.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(e)
                self.load_state_dict(state_dict, strict=False)

    def exclude_eval_metrics(self, checkpoint):
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('eval_metrics'):
                del checkpoint['state_dict'][key]

    def convert_state_dict(self, state_dict):
        for old_k in list(state_dict.keys()):
            new_k = old_k.replace('to_q.lora_layer', 'processor.to_q_lora')
            new_k = new_k.replace('to_k.lora_layer', 'processor.to_k_lora')
            new_k = new_k.replace('to_v.lora_layer', 'processor.to_v_lora')
            new_k = new_k.replace('to_out.0.lora_layer', 'processor.to_out_lora')
            state_dict[new_k] = state_dict.pop(old_k)

    def on_load_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)
        self.convert_state_dict(checkpoint['state_dict'])

    def on_save_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)

    def load_shared(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.hparams.model_id, subfolder="tokenizer", torch_dtype=torch.float16, use_safetensors=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.hparams.model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        self.text_encoder.requires_grad_(False)

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.hparams.model_id, subfolder="tokenizer_2", torch_dtype=torch.float16, use_safetensors=True)
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.hparams.model_id, subfolder="text_encoder_2", torch_dtype=torch.float16)
        self.text_encoder_2.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self.hparams.model_id, subfolder="vae", torch_dtype=torch.float16, use_safetensors=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae = torch.compile(self.vae)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.hparams.model_id, subfolder="scheduler", torch_dtype=torch.float16, use_safetensors=True)

    def add_lora(self, unet):
        """Adds LoRA adapter to the UNet using PEFT/inject_adapter_in_model."""
        lora_rank = self.hparams.lora_rank
        print(f"Attempting to add LoRA adapter with rank {lora_rank} using PEFT injection...")

        if not USE_PEFT_BACKEND or LoraConfig is None or inject_adapter_in_model is None:
             raise RuntimeError(
                 "PEFT backend is required for this LoRA implementation but is not available or enabled. "
                 "Please install PEFT (`pip install peft`) and ensure diffusers detects it."
             )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        print(f"Created LoraConfig: {lora_config}")

        try:
            inject_adapter_in_model(lora_config, unet, adapter_name="default")
            print("Successfully injected LoRA adapter using inject_adapter_in_model.")
        except Exception as e:
            print(f"Error during inject_adapter_in_model: {e}")
            print("Check PEFT installation, diffusers version compatibility, LoraConfig target_modules, and model structure.")
            raise e

        unet.requires_grad_(False)

        lora_params = []
        for name, param in unet.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                lora_params.append(param)

        if not lora_params:
            raise ValueError("No LoRA parameters found with 'lora_' in name after using inject_adapter_in_model(). Check LoraConfig target_modules.")
        else:
            print(f"Found and unfroze {len(lora_params)} LoRA parameters.")

        return (lora_params, 1.0)

    def get_cn(self, unet):
        cn = ControlNetModel.from_unet(unet)
        cn.enable_xformers_memory_efficient_attention()
        cn.enable_gradient_checkpointing()
        return cn, (list(cn.parameters()), 0.1)

    def load_branch(self, add_lora, train_lora, add_cn):
        unet = UNet2DConditionModel.from_pretrained(
            self.hparams.model_id, subfolder="unet", torch_dtype=torch.float32, use_safetensors=True)
        unet.enable_xformers_memory_efficient_attention()


        if add_cn:
            cn, params = self.get_cn(unet)
            self.trainable_params.append(params)
        else:
            cn = None

        if add_lora:
            unet_lora_params, lr_scale = self.add_lora(unet)
            params = (unet_lora_params, lr_scale) # Pack params tuple

            if train_lora and not add_cn:
                self.trainable_params.append(params)
                print(f"Added {len(unet_lora_params)} LoRA parameters to trainable list.")
            elif train_lora and add_cn:
                 print("NOTE: ControlNet is enabled ('add_cn' is True), LoRA parameters might not be added to optimizer if ControlNet params take precedence.")

        unet.enable_gradient_checkpointing()

        unet = torch.compile(unet)
        return unet, cn

    def load_pano(self):
        return self.load_branch(
            self.hparams.pano_lora,
            self.hparams.train_pano_lora,
            self.hparams.layout_cond,
        )

    def load_pers(self):
        return self.load_branch(
            self.hparams.pers_lora,
            self.hparams.train_pers_lora,
            self.hparams.layout_cond and self.hparams.pers_layout_cond,
        )

    @abstractmethod
    def instantiate_model(self):
        pass

    @torch.no_grad()
    def encode_text(self, text):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        prompt_embeds_1 = self.text_encoder(text_input_ids, output_hidden_states=True)
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2].to(self.dtype) # Use penultimate layer

        text_inputs_2 = self.tokenizer_2(
            text, padding="max_length", max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.text_encoder_2.device)
        outputs_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)
        prompt_embeds_2 = outputs_2.hidden_states[-2].to(self.dtype) # Use penultimate layer
        pooled_prompt_embeds = outputs_2.pooler_output.to(self.dtype)

        return prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds

    def _compute_time_ids(self, original_size, crops_coords_top_left, target_size, device):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype, device=device)
        return add_time_ids

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]
        original_model_dtype = self.dtype
        vae_original_dtype = vae.dtype

        x_input = rearrange(x_input.to(torch.float32), 'b l c h w -> (b l) c h w')

        if vae.dtype != torch.float32:
            vae.to(torch.float32)

        try:
            z = vae.encode(x_input).latent_dist
            z = z.sample()

            z = z * vae.config.scaling_factor

        finally:
            if vae.dtype != vae_original_dtype:
                vae.to(vae_original_dtype)

        z = z.to(original_model_dtype)
        z = rearrange(z, '(b l) c h w -> b l c h w', b=b)

        return z

    def pad_pano(self, pano, latent=False):
        b, m = pano.shape[:2]
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return pad_pano(pano, padding=padding)

    def unpad_pano(self, pano_pad, latent=False):
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return unpad_pano(pano_pad, padding=padding)

    def gen_cls_free_guide_pair(self, *inputs):
        result = []
        for input_val in inputs:
            if input_val is None:
                result.append(None)
            elif isinstance(input_val, dict):
                cfg_dict = {}
                for k, v in input_val.items():
                    if isinstance(v, torch.Tensor):
                        cfg_dict[k] = torch.cat([v] * 2)
                    else:
                        cfg_dict[k] = v
                result.append(cfg_dict)
            elif isinstance(input_val, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in input_val):
                result.append(type(input_val)(torch.cat([v]*2) for v in input_val))
            elif isinstance(input_val, torch.Tensor):
                result.append(torch.cat([input_val]*2))
            else:
                result.append(torch.cat([input_val]*2) if isinstance(input_val, torch.Tensor) else input_val)

        return result

    def combine_cls_free_guide_pred(self, *noise_pred_list):
        result = []
        for noise_pred in noise_pred_list:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.hparams.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)
            result.append(noise_pred)
        if len(result) == 1:
            return result[0]
        return result

    def rotate_latent(self, pano_latent, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent
        return torch.roll(pano_latent, int(degree / 360 * pano_latent.shape[-1]), dims=-1)

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b = latents.shape[0]
        if latents.ndim == 5:
            m = latents.shape[1]
            latents_reshaped = rearrange(latents, 'b m c h w -> (b m) c h w', b=b, m=m)
            batch_like_dim = b * m
        else:
            latents_reshaped = latents
            batch_like_dim = b
            m = 1

        original_model_dtype = self.dtype
        vae_original_dtype = vae.dtype
        latents_fp32 = latents_reshaped.to(torch.float32)

        scaling_factor = vae.config.scaling_factor
        latents_scaled = latents_fp32 / scaling_factor

        if vae.dtype != torch.float32:
            vae.to(torch.float32)

        image = None
        try:
            image = vae.decode(latents_scaled).sample

        finally:
            if vae.dtype != vae_original_dtype:
                vae.to(vae_original_dtype)

        image = image.to(original_model_dtype)

        if latents.ndim == 5:
            image = rearrange(image, '(b m) c h w -> b m c h w', b=b, m=m)

        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        if self.hparams.layout_cond:
            return optimizer
        else:
            scheduler = {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7),
                'interval': 'epoch', 
                'name': 'cosine_annealing_lr',
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output_dir = os.path.join(self.logger.save_dir, 'test', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dir = os.path.join(self.logger.save_dir, 'predict', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir, 'jpg')

    @abstractmethod
    def inference_and_save(self, batch, output_dir, ext):
        pass
