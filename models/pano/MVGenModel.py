import torch
import torch.nn as nn
from .modules import WarpAttn
from einops import rearrange
from utils.pano import pad_pano, unpad_pano


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, pano_unet, pers_cn=None, pano_cn=None, pano_pad=True):
        super().__init__()

        self.unet = unet
        self.pano_unet = pano_unet
        self.pers_cn = pers_cn
        self.pano_cn = pano_cn
        self.pano_pad = pano_pad

        if self.unet is not None:
            self.cp_blocks_encoder = nn.ModuleList()
            for downsample_block in self.unet.down_blocks:
                if hasattr(downsample_block, 'downsamplers') and downsample_block.downsamplers:
                    out_channels = downsample_block.downsamplers[-1].out_channels
                    self.cp_blocks_encoder.append(WarpAttn(out_channels))

            mid_channels = self.unet.mid_block.resnets[-1].out_channels
            self.cp_blocks_mid = WarpAttn(mid_channels)

            self.cp_blocks_decoder = nn.ModuleList()
            for upsample_block in self.unet.up_blocks:
                if hasattr(upsample_block, 'upsamplers') and upsample_block.upsamplers:
                    in_channels = upsample_block.upsamplers[0].channels
                    self.cp_blocks_decoder.append(WarpAttn(in_channels))

            self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
                list(self.cp_blocks_decoder.parameters()) + \
                list(self.cp_blocks_encoder.parameters()), 1.0)]

    def forward(self, pers_latents, pano_latent, timestep,
                pers_prompt_embd, pers_prompt_embd_2, pers_pooled_prompt_embd, # Perspective embeddings
                pano_prompt_embd, pano_prompt_embd_2, pano_pooled_prompt_embd, # Panoramic embeddings
                add_time_ids, # Tuple: (pers_time_ids, pano_time_ids)
                cameras,
                pers_layout_cond=None, pano_layout_cond=None):

        pers_timestep, pano_timestep = timestep if isinstance(timestep, (tuple, list)) else (timestep, timestep)
        pers_add_time_ids, pano_add_time_ids = add_time_ids if isinstance(add_time_ids, (tuple, list)) else (add_time_ids, add_time_ids)

        pers_encoder_hidden_states = None
        pers_hidden_states = None
        pers_emb = None
        pers_added_cond_kwargs = {}
        pers_b_times_m = None

        if pers_latents is not None and self.unet is not None:
            b, m, c, h, w = pers_latents.shape
            pers_b_times_m = b * m

            pers_hidden_states = rearrange(pers_latents, 'b m c h w -> (b m) c h w')

            _pers_prompt_embd = rearrange(pers_prompt_embd, 'b m l c -> (b m) l c') if pers_prompt_embd is not None else None
            _pers_prompt_embd_2 = rearrange(pers_prompt_embd_2, 'b m l c -> (b m) l c') if pers_prompt_embd_2 is not None else None
            _pers_pooled_prompt_embd = rearrange(pers_pooled_prompt_embd, 'b m c -> (b m) c') if pers_pooled_prompt_embd is not None else None

            if _pers_prompt_embd is not None and _pers_prompt_embd_2 is not None:
                pers_encoder_hidden_states = torch.cat([_pers_prompt_embd, _pers_prompt_embd_2], dim=-1)
            else:
                pers_encoder_hidden_states = _pers_prompt_embd # Or handle appropriately if one is None

            if cameras is not None:
                 cameras_reshaped = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
            else:
                 cameras_reshaped = None

            if pers_timestep.ndim == 1 and pers_timestep.shape[0] == b: 
                 _pers_timestep = pers_timestep[:, None].repeat(1, m).reshape(-1) 
            elif pers_timestep.ndim == 1 and pers_timestep.shape[0] == b * m:
                 _pers_timestep = pers_timestep
            elif pers_timestep.ndim == 2 and pers_timestep.shape == (b, m):
                 _pers_timestep = pers_timestep.reshape(-1)
            elif pers_timestep.ndim == 0:
                 _pers_timestep = pers_timestep.repeat(b*m)
            else:
                 if pers_timestep.numel() * m == pers_b_times_m:
                      _pers_timestep = pers_timestep.repeat_interleave(m)
                 else:
                      raise ValueError(f"Unexpected pers_timestep shape: {pers_timestep.shape} for b={b}, m={m}")


            pers_t_emb = self.unet.time_proj(_pers_timestep).to(dtype=self.unet.dtype)
            pers_emb = self.unet.time_embedding(pers_t_emb)
            pers_added_cond_kwargs = {"text_embeds": _pers_pooled_prompt_embd, "time_ids": pers_add_time_ids}

        pano_b = pano_latent.shape[0]
        pano_m = pano_latent.shape[1] 
        assert pano_m == 1, f"Expected pano_latent to have 1 view dim, but got {pano_m}"
        pano_hidden_states = rearrange(pano_latent, 'b m c h w -> (b m) c h w', m=pano_m)

        _pano_prompt_embd = rearrange(pano_prompt_embd, 'b m l c -> (b m) l c', m=pano_m) if pano_prompt_embd is not None else None
        _pano_prompt_embd_2 = rearrange(pano_prompt_embd_2, 'b m l c -> (b m) l c', m=pano_m) if pano_prompt_embd_2 is not None else None
        _pano_pooled_prompt_embd = rearrange(pano_pooled_prompt_embd, 'b m c -> (b m) c', m=pano_m) if pano_pooled_prompt_embd is not None else None

        if _pano_prompt_embd is not None and _pano_prompt_embd_2 is not None:
            pano_encoder_hidden_states = torch.cat([_pano_prompt_embd, _pano_prompt_embd_2], dim=-1)
        else:
            # print("Warning: Both panoramic text embeddings not available for concatenation.")
            pano_encoder_hidden_states = _pano_prompt_embd

        if pano_timestep.ndim == 0: 
            _pano_timestep = pano_timestep.repeat(pano_b)
        elif pano_timestep.shape == (pano_b,):
            _pano_timestep = pano_timestep
        else:
             if pano_timestep.numel() == pano_b:
                 _pano_timestep = pano_timestep.reshape(-1)
             else:
                 raise ValueError(f"Unexpected pano_timestep shape: {pano_timestep.shape} for pano_b={pano_b}")


        pano_t_emb = self.pano_unet.time_proj(_pano_timestep).to(dtype=self.pano_unet.dtype)
        pano_emb = self.pano_unet.time_embedding(pano_t_emb)
        pano_added_cond_kwargs = {"text_embeds": _pano_pooled_prompt_embd, "time_ids": pano_add_time_ids}

        # === 1. ControlNet Processing ===
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if self.pers_cn is not None and pers_hidden_states is not None:
            _pers_layout_cond = None
            if pers_layout_cond is not None:
                if pers_layout_cond.shape[0] == b and pers_layout_cond.shape[1] == m:
                    _pers_layout_cond = rearrange(pers_layout_cond, 'b m ... -> (b m) ...')
                elif pers_layout_cond.shape[0] == pers_b_times_m:
                     _pers_layout_cond = pers_layout_cond
                else:
                     raise ValueError(f"Unexpected pers_layout_cond shape {pers_layout_cond.shape}")

            down_block_additional_residuals, mid_block_additional_residual = self.pers_cn(
                pers_hidden_states, 
                _pers_timestep,
                encoder_hidden_states=pers_encoder_hidden_states,
                added_cond_kwargs=pers_added_cond_kwargs,
                controlnet_cond=_pers_layout_cond,
                return_dict=False,
            )
        pano_down_block_additional_residuals = None
        pano_mid_block_additional_residual = None
        if self.pano_cn is not None:
            _pano_layout_cond = pano_layout_cond
            if _pano_layout_cond is not None and len(_pano_layout_cond.shape) > 4:
                 _pano_layout_cond = rearrange(_pano_layout_cond, 'b 1 c h w -> b c h w')
            elif _pano_layout_cond is not None and _pano_layout_cond.shape[0] != pano_b:
                 raise ValueError("Unexpected pano_layout_cond shape")

            pano_down_block_additional_residuals, pano_mid_block_additional_residual = self.pano_cn(
                pano_hidden_states,
                _pano_timestep,
                encoder_hidden_states=pano_encoder_hidden_states,
                added_cond_kwargs=pano_added_cond_kwargs,
                controlnet_cond=_pano_layout_cond,
                return_dict=False,
            )

        # === 2. Initial Convolution ===
        if pers_hidden_states is not None:
            pers_hidden_states = self.unet.conv_in(pers_hidden_states)

        if self.pano_pad:
            pano_hidden_states_padded = pad_pano(pano_hidden_states, 1)
            pano_hidden_states = self.pano_unet.conv_in(pano_hidden_states_padded)
            pano_hidden_states = unpad_pano(pano_hidden_states, 1)
        else:
            pano_hidden_states = self.pano_unet.conv_in(pano_hidden_states)

        # === 3. UNet Down Blocks ===
        pers_down_block_res_samples = (pers_hidden_states,) if pers_hidden_states is not None else None
        pano_down_block_res_samples = (pano_hidden_states,)
        num_down_blocks = len(self.pano_unet.down_blocks)
        encoder_warp_block_idx = 0

        for i in range(num_down_blocks):
            pers_block = self.unet.down_blocks[i] if self.unet is not None and i < len(self.unet.down_blocks) else None
            pano_block = self.pano_unet.down_blocks[i]
            num_layers = len(pano_block.resnets)

            for j in range(num_layers):
                if pers_block is not None and j < len(pers_block.resnets) and pers_hidden_states is not None:
                    pers_res_input = pers_hidden_states
                    pers_hidden_states = pers_block.resnets[j](pers_hidden_states, pers_emb)
                    if hasattr(pers_block, 'attentions') and pers_block.attentions and j < len(pers_block.attentions):
                         pers_hidden_states = pers_block.attentions[j](
                             pers_hidden_states,
                             encoder_hidden_states=pers_encoder_hidden_states,
                             added_cond_kwargs=pers_added_cond_kwargs
                         ).sample
                    if down_block_additional_residuals is not None and len(down_block_additional_residuals) > 0:
                         pers_hidden_states = pers_hidden_states + down_block_additional_residuals.pop(0)
                    if pers_down_block_res_samples is not None: 
                       pers_down_block_res_samples += (pers_hidden_states,)

                pano_res_input = pano_hidden_states
                if self.pano_pad: pano_hidden_states = pad_pano(pano_hidden_states, 2)
                pano_hidden_states = pano_block.resnets[j](pano_hidden_states, pano_emb)
                if hasattr(pano_block, 'attentions') and pano_block.attentions and j < len(pano_block.attentions):
                    pano_hidden_states = pano_block.attentions[j](
                        pano_hidden_states,
                        encoder_hidden_states=pano_encoder_hidden_states,
                        added_cond_kwargs=pano_added_cond_kwargs
                    ).sample
                if self.pano_pad: pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                if pano_down_block_additional_residuals is not None and len(pano_down_block_additional_residuals) > 0:
                    pano_hidden_states = pano_hidden_states + pano_down_block_additional_residuals.pop(0)
                pano_down_block_res_samples += (pano_hidden_states,)

            has_pers_downsampler = pers_block is not None and hasattr(pers_block, 'downsamplers') and pers_block.downsamplers
            has_pano_downsampler = hasattr(pano_block, 'downsamplers') and pano_block.downsamplers

            if has_pano_downsampler:
                if has_pers_downsampler and pers_hidden_states is not None:
                     pers_hidden_states = pers_block.downsamplers[0](pers_hidden_states)
                     if pers_down_block_res_samples is not None:
                         pers_down_block_res_samples += (pers_hidden_states,)

                if self.pano_pad: pano_hidden_states = pad_pano(pano_hidden_states, 2)
                pano_hidden_states = pano_block.downsamplers[0](pano_hidden_states)
                if self.pano_pad: pano_hidden_states = unpad_pano(pano_hidden_states, 1)
                pano_down_block_res_samples += (pano_hidden_states,)

                if pers_hidden_states is not None and encoder_warp_block_idx < len(self.cp_blocks_encoder) and cameras_reshaped is not None:
                    pers_hidden_states, pano_hidden_states = self.cp_blocks_encoder[encoder_warp_block_idx](
                        pers_hidden_states, pano_hidden_states, cameras_reshaped) # Use reshaped cameras
                    encoder_warp_block_idx += 1


        # === 4. UNet Mid Block ===
        if self.unet is not None and self.unet.mid_block is not None and pers_hidden_states is not None:
            pers_hidden_states = self.unet.mid_block.resnets[0](pers_hidden_states, pers_emb)
            num_mid_attns = len(self.unet.mid_block.attentions)
            num_mid_resnets = len(self.unet.mid_block.resnets[1:])
            for idx in range(max(num_mid_attns, num_mid_resnets)):
                 if idx < num_mid_attns:
                     pers_hidden_states = self.unet.mid_block.attentions[idx](
                         pers_hidden_states,
                         encoder_hidden_states=pers_encoder_hidden_states,
                         added_cond_kwargs=pers_added_cond_kwargs
                     ).sample
                 if idx < num_mid_resnets:
                     pers_hidden_states = self.unet.mid_block.resnets[1+idx](pers_hidden_states, pers_emb)

            if mid_block_additional_residual is not None:
                pers_hidden_states = pers_hidden_states + mid_block_additional_residual

        if self.pano_pad: pano_hidden_states = pad_pano(pano_hidden_states, 2)
        pano_hidden_states = self.pano_unet.mid_block.resnets[0](pano_hidden_states, pano_emb)
        num_mid_attns_pano = len(self.pano_unet.mid_block.attentions)
        num_mid_resnets_pano = len(self.pano_unet.mid_block.resnets[1:])
        for idx in range(max(num_mid_attns_pano, num_mid_resnets_pano)):
             if idx < num_mid_attns_pano:
                 pano_hidden_states = self.pano_unet.mid_block.attentions[idx](
                     pano_hidden_states,
                     encoder_hidden_states=pano_encoder_hidden_states,
                     added_cond_kwargs=pano_added_cond_kwargs
                 ).sample
             if idx < num_mid_resnets_pano:
                 pano_hidden_states = self.pano_unet.mid_block.resnets[1+idx](pano_hidden_states, pano_emb)

        if self.pano_pad: pano_hidden_states = unpad_pano(pano_hidden_states, 2)
        if pano_mid_block_additional_residual is not None:
            pano_hidden_states = pano_hidden_states + pano_mid_block_additional_residual

        if pers_hidden_states is not None and self.cp_blocks_mid is not None and cameras_reshaped is not None:
            pers_hidden_states, pano_hidden_states = self.cp_blocks_mid(
                pers_hidden_states, pano_hidden_states, cameras_reshaped) # Use reshaped cameras


        # === 5. UNet Up Blocks ===
        num_up_blocks = len(self.pano_unet.up_blocks)
        decoder_warp_block_idx = 0 # Counter for decoder warp blocks

        for i in range(num_up_blocks):
            pers_block = self.unet.up_blocks[i] if self.unet is not None and i < len(self.unet.up_blocks) else None
            pano_block = self.pano_unet.up_blocks[i]

            pano_skip_count = len(pano_block.resnets)
            pers_skip_count = pano_skip_count if pers_block else 0

            pers_res_samples = None
            if pers_block is not None and pers_hidden_states is not None and pers_down_block_res_samples is not None:
                 if len(pers_down_block_res_samples) >= pers_skip_count:
                      pers_res_samples = pers_down_block_res_samples[-pers_skip_count:]
                      pers_down_block_res_samples = pers_down_block_res_samples[:-pers_skip_count]
                 else:
                      print(f"Warning: Not enough pers skip samples for up block {i}. Need {pers_skip_count}, have {len(pers_down_block_res_samples)}.")
                      pers_res_samples = []

            if len(pano_down_block_res_samples) >= pano_skip_count:
                pano_res_samples = pano_down_block_res_samples[-pano_skip_count:]
                pano_down_block_res_samples = pano_down_block_res_samples[:-pano_skip_count]
            else:
                 print(f"Warning: Not enough pano skip samples for up block {i}. Need {pano_skip_count}, have {len(pano_down_block_res_samples)}.")
                 pano_res_samples = []

            has_pers_upsampler = pers_block is not None and hasattr(pers_block, 'upsamplers') and pers_block.upsamplers
            has_pano_upsampler = hasattr(pano_block, 'upsamplers') and pano_block.upsamplers

            num_layers = len(pano_block.resnets)
            for j in range(num_layers):
                if pers_block is not None and j < len(pers_block.resnets) and pers_hidden_states is not None and pers_res_samples and j < len(pers_res_samples):
                    pers_res_hidden_states = pers_res_samples[-1 - j]
                    pers_hidden_states = torch.cat([pers_hidden_states, pers_res_hidden_states], dim=1)
                    pers_hidden_states = pers_block.resnets[j](pers_hidden_states, pers_emb)
                    if hasattr(pers_block, 'attentions') and pers_block.attentions and j < len(pers_block.attentions):
                        pers_hidden_states = pers_block.attentions[j](
                            pers_hidden_states,
                            encoder_hidden_states=pers_encoder_hidden_states,
                            added_cond_kwargs=pers_added_cond_kwargs
                        ).sample

                if pano_res_samples and j < len(pano_res_samples):
                    pano_res_hidden_states = pano_res_samples[-1 - j]
                    pano_hidden_states = torch.cat([pano_hidden_states, pano_res_hidden_states], dim=1)
                    if self.pano_pad: pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = pano_block.resnets[j](pano_hidden_states, pano_emb)
                    if hasattr(pano_block, 'attentions') and pano_block.attentions and j < len(pano_block.attentions):
                        pano_hidden_states = pano_block.attentions[j](
                            pano_hidden_states,
                            encoder_hidden_states=pano_encoder_hidden_states,
                            added_cond_kwargs=pano_added_cond_kwargs
                        ).sample
                    if self.pano_pad: pano_hidden_states = unpad_pano(pano_hidden_states, 2)

            if has_pano_upsampler and pers_hidden_states is not None and decoder_warp_block_idx < len(self.cp_blocks_decoder) and cameras_reshaped is not None:
                pers_hidden_states, pano_hidden_states = self.cp_blocks_decoder[decoder_warp_block_idx](
                    pers_hidden_states, pano_hidden_states, cameras_reshaped)
                decoder_warp_block_idx += 1

            if has_pano_upsampler:
                if has_pers_upsampler and pers_hidden_states is not None:
                    pers_hidden_states = pers_block.upsamplers[0](pers_hidden_states)

                if self.pano_pad: pano_hidden_states = pad_pano(pano_hidden_states, 1)
                pano_hidden_states = pano_block.upsamplers[0](pano_hidden_states)
                if self.pano_pad: pano_hidden_states = unpad_pano(pano_hidden_states, 2)


        # === 6. Post-process ===
        pers_sample = None
        if self.unet is not None and pers_hidden_states is not None:
            pers_sample = self.unet.conv_norm_out(pers_hidden_states)
            pers_sample = self.unet.conv_act(pers_sample)
            pers_sample = self.unet.conv_out(pers_sample)
            original_b = pers_latents.shape[0]
            pers_sample = rearrange(pers_sample, '(b m) c h w -> b m c h w', b=original_b) 

        pano_sample = self.pano_unet.conv_norm_out(pano_hidden_states)
        pano_sample = self.pano_unet.conv_act(pano_sample)
        if self.pano_pad:
            pano_sample = pad_pano(pano_sample, 1)
        pano_sample = self.pano_unet.conv_out(pano_sample)
        if self.pano_pad:
            pano_sample = unpad_pano(pano_sample, 1)

        pano_sample = rearrange(pano_sample, 'b c h w -> b 1 c h w', b=pano_b)

        return pers_sample, pano_sample