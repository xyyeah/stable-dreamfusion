from functools import partial

import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torch import TensorType
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock, TimestepBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf.listconfig import ListConfig

from cldm.triplane import TriplaneGenerator, TriplaneGenerator2, TriplaneGenerator3, TriplaneGenerator4, \
    TriplaneGenerator5, TriplaneGenerator6, TriplaneGenerator7, TriplaneGenerator8, TriplaneGenerator9, \
    TriplaneGenerator10, TriplaneGenerator11, TriplaneGenerator12, TriplaneGenerator13
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.model_components.ray_samplers import UniformSampler, PDFSampler, SpacedSampler
from nerfstudio.model_components.renderers import RGBRenderer, AccumulationRenderer
from nerfstudio.model_components.scene_colliders import SceneCollider, AABBBoxCollider
# AABBBoxCollider, SphereCollider, NearFarCollider,
from nerfstudio.utils import colors
from nerfacc import ContractionType, contract
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_triplane_encoder(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_triplane_encoder(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_triplane_encoder(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_triplane_encoder(ch)
        self._feature_size += ch

        self.render = Render(model_channels=model_channels, time_embed_dim=time_embed_dim,
                             hint_channels=hint_channels, dropout=dropout)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def make_triplane_encoder(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
        # return Render(model_channels=channels, dims=self.dims)

    def get_buffered_grid(self, x):
        if not hasattr(self, 'buffered_grid') or x.size(0) > self.buffered_grid.size(0):
            setattr(self, 'buffered_grid', get_grid(x.size(0), (64, 64, 64)))
        return self.buffered_grid[:x.size(0)]

    def forward(self, x, hint, pose, intrinsic, dist, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # pose_inv = pose.clone().to(x.device)
        # pose_inv[:, :3, :3] = pose[:, :3, :3].transpose(-1, -2).to(x.device)
        # pose_inv[:, :3, :3] = torch.eye(3).expand(x.size(0), 1, 1)
        outs = []
        # h = torch.cat([x, hint], dim=1).type(self.dtype)
        h = x.type(self.dtype)

        guided_hint = True
        render_rgb = None
        for module, render in zip(self.input_blocks, self.zero_convs):
            if guided_hint:
                render_hint, render_rgb = self.render(hint, pose=pose, intrinsic=intrinsic, dist=dist)
                h = module(h, emb, context)
                h = h + render_hint
                guided_hint = False
            else:
                h = module(h, emb, context)
            outs.append(render(h, pose))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, pose))
        return {
            'control': outs,
            'render_rgb': render_rgb
        }


def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    r"""Get a grid ranging [-1, 1] of 2D/3D coordinates.

    Args:
        batchsize (int) : Batch size.
        size (tuple) : (height, width) or (depth, height, width).
        minval (float) : minimum value in returned grid.
        maxval (float) : maximum value in returned grid.
    Returns:
        t_grid (4D tensor) : Grid of coordinates.
    """
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError('Dimension can only be 2 or 3.')
    x = torch.linspace(minval, maxval, cols)
    x = x.view(1, 1, 1, cols)
    x = x.expand(batchsize, 1, rows, cols)

    y = torch.linspace(minval, maxval, rows)
    y = y.view(1, 1, rows, 1)
    y = y.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([x, y], dim=1)

    if len(size) == 3:
        z = torch.linspace(minval, maxval, deps)
        z = z.view(1, 1, deps, 1, 1)
        z = z.expand(batchsize, 1, deps, rows, cols)

        t_grid = t_grid.unsqueeze(2).expand(batchsize, 2, deps, rows, cols)
        t_grid = torch.cat([t_grid, z], dim=1)

    t_grid.requires_grad = False
    return t_grid


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        # ray = batch['ray']
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        encoder_posterior = self.encode_first_stage(control)
        control = self.get_first_stage_encoding(encoder_posterior).detach()

        return x, dict(c_crossattn=[c], c_concat=[control],
                       pose=batch['pose'].cpu(), intrinsic=batch["intrinsic"].cpu(), dist=batch["dist"].cpu())

    def apply_model(self, x_noisy, t, cond, return_rgb=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_img = None

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            outs = self.control_model(x=x_noisy, hint=hint, pose=cond['pose'], intrinsic=cond["intrinsic"],
                                      dist=cond["dist"].cpu(),
                                      timesteps=t, context=cond_txt)
            control = outs['control']
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt,
                                  control=control, only_mid_control=self.only_mid_control)
        # eps2 = self.predict_noise_from_start(x_noisy, t, outs['render_rgb'])
        eps2 = 0.0
        if return_rgb:
            return eps + eps2, outs['render_rgb']
        return eps + eps2

    def p_losses(self, x_start, cond, t, noise=None):
        from ldm.util import default
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, render_rgb = self.apply_model(x_noisy, t, cond, return_rgb=True)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        loss_recon = self.get_loss(render_rgb, x_start, mean=True)
        loss += loss_recon
        loss_dict.update({f'{prefix}/loss_recon': loss_recon})

        return loss, loss_dict

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        N = min(z.shape[0], N)
        guided_hint = c["c_concat"][0][:N]
        c_cross = c["c_crossattn"][0][:N]
        # ray = batch['ray']
        # guided_rgb, guided_hint = self.control_model.proj(guided_hint, ray, batch['camera'])
        c_cat = guided_hint
        # c["c_rgb"] = guided_rgb
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = batch[self.control_key][:N].permute(0, 3, 1, 2)

        # log["view"] = self.decode_first_stage(
        #     self.control_model(x=None, hint=torch.cat(c['c_concat'], 1), pose=c['pose'],
        #                        timesteps=None, context=None, only_get_hint=True)['render_rgb']
        # )

        # log["view"] = self.decode_first_stage(guided_rgb)
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat],
                                                           "c_crossattn": [c_cross],
                                                           'pose': c['pose'],
                                                           'intrinsic': c["intrinsic"],
                                                           'dist': c['dist']},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            # from data.datasets.panorama_datasets import concat_ray_bundle
            # ray = concat_ray_bundle(ray, ray)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross],
                       'pose': c['pose'], 'intrinsic': c["intrinsic"], 'dist': c['dist']}
            samples_cfg, intermediates = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c_cross],
                      'pose': c['pose'], 'intrinsic': c["intrinsic"], 'dist': c['dist']},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )

            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            log['view'] = self.decode_first_stage(intermediates['rgb'][-1])

        result_grid = make_grid(
            torch.cat([
                log["control"],
                log["reconstruction"],
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"],
                log["view"],
            ], dim=0), nrow=N)
        log = {'result': result_grid}
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 1, w // 1)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.control_model.parameters())
        # if not self.sd_locked:
        #     params += list(self.model.diffusion_model.output_blocks.parameters())
        #     params += list(self.model.diffusion_model.out.parameters())
        p1, p2 = {'params': [], 'lr': self.learning_rate}, {'params': [], 'lr': self.learning_rate * 10.0}
        for n, p in self.control_model.named_parameters():
            if 'render' in n or 'control_img_proj' in n:
                p2['params'].append(p)
            else:
                p1['params'].append(p)
        opt = torch.optim.AdamW([p1, p2], lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_ƒmodel = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class Identity:
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class Log:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log(x)


class Exp:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.exp(x)


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, near_plane: float, far_plane: float, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        near_plane = self.near_plane
        ray_bundle.nears = ones * near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle


class Render(nn.Module):
    def __init__(self, hint_channels, time_embed_dim, model_channels, num_samples_per_ray=24, dropout=0.0,
                 dims=2):
        super().__init__()

        self.tri_generator = TriplaneGenerator13(hint_channels, model_channels, time_embed_dim,
                                                 dropout=dropout, channel_mult=(1, 2, 2, 4))

        self.dims = dims

        self.direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.sampler = SpacedSampler(
            spacing_fn=Identity(), spacing_fn_inv=Identity(),
            num_samples=num_samples_per_ray, single_jitter=False, train_stratified=True)

        aabb_scale = 1.0
        self.scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1 * aabb_scale, -1 * aabb_scale, -1 * aabb_scale],
                 [1 * aabb_scale, 1 * aabb_scale, 1 * aabb_scale]],
                dtype=torch.float32
            )
        )
        self.collider = AABBBoxCollider2(scene_box=self.scene_box, near_plane=0.01)

        self.model_channels = model_channels
        self.mlp_head = nn.Sequential(
            conv_nd(self.dims, self.direction_encoding.get_out_dim() + model_channels, model_channels, 1,
                    padding=0),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, model_channels, model_channels, 1, padding=0)),
        )

        self.model_channels = model_channels

        self.map2rgb = nn.Sequential(
            # normalization(model_channels),
            conv_nd(self.dims, model_channels, model_channels, 1, padding=0),
            nn.SiLU(),
            conv_nd(self.dims, model_channels, model_channels, 1, padding=0),
            nn.SiLU(),
            conv_nd(self.dims, model_channels, hint_channels, 1, padding=0),
        )

    def generate_rays(self, cameras, down=1):
        if not hasattr(self, 'img_coords'):
            img_coords = cameras.get_image_coords()
            height, width = int(cameras.height[0][0]), int(cameras.width[0][0])
            # print(nc, c.size(), cameras.shape, cameras.camera_to_worlds.size())
            # idx1 = torch.arange(0, end=height - 1, step=down).long()
            idx1 = torch.linspace(0, height - 1, steps=(height // down)).long()
            idx2 = torch.linspace(0, width - 1, steps=(width // down)).long()
            img_coords = img_coords[idx1][:, idx2]
            setattr(self, 'img_coords', img_coords)
        nc = cameras.camera_to_worlds.size(0)
        img_coords = self.img_coords
        c = torch.arange(0, nc)[:, None, None].expand(-1, img_coords.size(0), img_coords.size(1)).reshape(-1)
        img_coords = img_coords[None].expand(nc, -1, -1, -1).reshape(-1, 2)
        ray_bundle = cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=img_coords
        )
        return ray_bundle

    @torch.no_grad()
    def sample_points(self, x, pose, intrinsic, dist):
        # import pdb; pdb.set_trace()
        pose = pose.type(x.dtype)
        intrinsic = intrinsic.type(x.dtype)
        B, _, H, W = x.size()
        H, W = 64, 64
        B = B // 3
        aabb_scale = 1.0
        ori_focal, ori_principle_point = intrinsic.chunk(chunks=2, dim=1)
        focal, resolution = ori_focal, ori_principle_point * 2
        res1, res2 = [], []
        for i in range(B):
            cameras = Cameras(
                fx=focal[i:i + 1],
                fy=focal[i:i + 1],
                cx=resolution[i:i + 1] // 2,
                cy=resolution[i:i + 1] // 2,
                distortion_params=None,
                height=resolution[i:i + 1].type(torch.IntTensor),
                width=resolution[i:i + 1].type(torch.IntTensor),
                camera_to_worlds=pose[i:i + 1],
                camera_type=CameraType.PERSPECTIVE,
            ).to(x.device)
            ray_bundle = self.generate_rays(cameras, 8)

            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-1 * aabb_scale, -1 * aabb_scale, -1 - float(dist[i])],
                     [1 * aabb_scale, 1 * aabb_scale, 1 - float(dist[i])]],
                    dtype=torch.float32
                )
            )
            collider = AABBBoxCollider2(scene_box=scene_box, near_plane=0.01)
            ray_bundle = collider(ray_bundle)
            ray_samples = self.sampler(ray_bundle)
            positions = ray_samples.frustums.get_positions()
            positions = SceneBox.get_normalized_positions(positions, scene_box.aabb.to(x.device)).detach()
            Np = positions.size(1)
            positions = positions * 2 - 1
            positions[..., 1] = -positions[..., 1]
            positions = positions.reshape(1, H * W, Np, 3)
            res1.append(positions)
            res2.append(ray_samples)
        return torch.cat(res1, dim=0), res2

    def forward(self, x, pose, intrinsic, dist):
        x = self.tri_generator(x, pad=0)  # yx, zx, zy

        self.sampler.num_samples = 24
        Np = self.sampler.num_samples
        B, _, H, W = x.size()
        H, W = 64, 64
        B = B // 3
        positions, ray_samples = self.sample_points(x, pose, intrinsic, dist)

        def get_density(field, coords):
            plane_features = F.grid_sample(
                field, coords, align_corners=True, mode='bilinear'
            )  # 3BxDx(HxW)xNp
            plane_features = torch.stack(plane_features.chunk(chunks=3, dim=0), dim=1).sum(1)
            plane_features = torch.relu_(plane_features.sum(dim=1))[:, :, :, None]
            return plane_features

        def get_rgb(field, coords):
            d = torch.cat([rs.frustums.directions for rs in ray_samples], dim=0)
            d_encoded = self.direction_encoding(d).reshape(B, H * W, Np, -1).permute(0, 3, 1, 2)
            plane_features = F.grid_sample(
                field, coords, align_corners=True, mode='bilinear'
            )  # 3BxDx(HxW)xNp
            plane_features = torch.stack(plane_features.chunk(chunks=3, dim=0), dim=1).sum(1)
            plane_features = self.mlp_head(torch.cat([plane_features, d_encoded], dim=1))
            plane_features = plane_features.permute(0, 2, 3, 1)
            return plane_features

        def get_field(x):
            color_field, density_field = torch.split(x, [self.model_channels, 16], dim=1)
            return color_field, density_field

        color_field, density_field = get_field(x)
        plane_coords = torch.cat([positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]], dim=0)
        density = get_density(density_field, plane_coords)  # .reshape(B * H * W, Np, 1)
        rgb = get_rgb(color_field, plane_coords)  # .reshape(B * H * W, Np, self.model_channels)
        weights = torch.stack([rs.get_weights(density[i]) for i, rs in enumerate(ray_samples)], dim=0)
        accumulated_weight = torch.sum(weights, dim=-2)  # B, H * W, D

        comp_rgb = torch.sum(weights * rgb, dim=-2)
        comp_rgb = comp_rgb + (1.0 - accumulated_weight) * torch.zeros_like(rgb[..., -1, :])
        feats = comp_rgb.reshape(B, H, W, self.model_channels).permute(0, 3, 1, 2).contiguous()
        rgb = self.map2rgb(feats)
        return feats, rgb


class AABBBoxCollider2(AABBBoxCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box: SceneBox, near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(scene_box, near_plane, **kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane

    def _intersect_with_aabb(
            self, rays_o, rays_d, aabb
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins
            rays_d: (num_rays, 3) ray directions
            aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        nears = torch.max(
            torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
        ).values
        fars = torch.min(
            torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
        ).values

        # import pdb; pdb.set_trace()
        # print(aabb)
        # print(rays_o)
        # print(dir_fraction)

        # clamp to near plane
        near_plane = self.near_plane
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)

        return nears, fars