import random
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
from ldm.util import default
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock, TimestepBlock, ResBlock2
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim2 import DDIMSampler
from omegaconf.listconfig import ListConfig

from cldm.triplane import TriplaneGenerator14, TriplaneGenerator15, TriplaneGenerator16, TriplaneAttention
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


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, hint=None,
                only_mid_control=False, y=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                label_emb = self.label_emb(y)
                emb = emb + label_emb

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)

        h = self.middle_block(h, emb, context) + control.pop()

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlLDM(LatentDiffusion):

    def __init__(self,
                 embedder_config, control_stage_config,
                 control_key, only_mid_control, embedding_key="jpg", embedding_dropout=0.5,
                 freeze_embedder=True, noise_aug_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 25
        self.embedding_dropout = embedding_dropout
        self._init_embedder(embedder_config, freeze_embedder)
        self._init_noise_aug(noise_aug_config)

    def _init_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def _init_noise_aug(self, config):
        if config is not None:
            # use the KARLO schedule for noise augmentation on CLIP image embeddings
            noise_augmentor = instantiate_from_config(config)
            assert isinstance(noise_augmentor, nn.Module)
            noise_augmentor = noise_augmentor.eval()
            noise_augmentor.train = disabled_train
            self.noise_augmentor = noise_augmentor
        else:
            self.noise_augmentor = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, noise_level=None, *args, **kwargs):
        bs, n_pose = batch['pose'].size(0), batch['pose'].size(1)
        control = batch[self.first_stage_key][:, 0]
        batch[self.first_stage_key] = torch.cat(batch[self.first_stage_key].unbind(dim=1), dim=0)

        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        c_adm = self.embedder(control)
        c_adm = torch.cat([c_adm for _ in range(n_pose)], dim=0)

        # if self.noise_augmentor is not None:
        #     c_adm, noise_level_emb = self.noise_augmentor(c_adm, noise_level)
        #     # assume this gives embeddings of noise levels
        #     c_adm = torch.cat((c_adm, noise_level_emb), 1)
        # if self.training:
        #     mask = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(c_adm.shape[0], device=c_adm.device))
        #     c_adm = mask[:, None] * c_adm

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if self.noise_augmentor is not None:
            c_adm, noise_level_emb = self.noise_augmentor(c_adm, noise_level)
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        if self.training:
            mask = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(c_adm.shape[0], device=c_adm.device))
            c_adm = mask[:, None] * c_adm

        encoder_posterior = self.encode_first_stage(control)
        control = self.get_first_stage_encoding(encoder_posterior).detach()

        return x, dict(
            c_crossattn=[c.repeat(n_pose, 1, 1)], c_adm=c_adm, c_concat=control,
            pose=batch['pose'].cpu(), intrinsic=batch["intrinsic"].cpu(), dist=batch["dist"].cpu()
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        c_adm = cond['c_adm']
        c_concat = cond['c_concat']

        control = self.control_model(
            x_noisy, t, cond_txt, ref_img=c_concat,
            pose=cond['pose'], intrinsic=cond["intrinsic"], dist=cond["dist"], y=c_adm
        )
        # print(control.size(), c_concat.size())
        # c_concat = self.decode_first_stage(c_concat)
        # control = self.decode_first_stage(control)
        # result_grid = make_grid(
        #     torch.cat([
        #         c_concat,
        #         control,
        #         # log["view"],
        #     ], dim=0), nrow=control.size(0))
        # result_grid = result_grid.float().detach().cpu()
        # result_grid = torch.clamp(result_grid, -1., 1.)
        # grid = (result_grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        # grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        # grid = grid.numpy()
        # import numpy as np
        # grid = (grid * 255).astype(np.uint8)
        # from PIL import Image
        # Image.fromarray(grid).save('/workspace/debug.png')
        # exit(0)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, y=c_adm,
                              control=control, only_mid_control=self.only_mid_control)
        return eps

    def forward(self, x, c, *args, **kwargs):
        N = c['pose'].size(1)
        t = torch.randint(0, self.num_timesteps, (c['pose'].size(0),), device=self.device).long()
        t = torch.cat([t for _ in range(N)], dim=0)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

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

        # loss_recon = self.get_loss(render_rgb,
        #                            x_start, mean=True)
        # loss += 5e-2 * loss_recon
        # loss_dict.update({f'{prefix}/loss_recon': loss_recon})

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
        z, c = self.get_input(batch, self.first_stage_key)
        N = z.shape[0]
        c_cross = c["c_crossattn"][0]
        c_adm = c["c_adm"]
        c_concat = c['c_concat']
        n_row = z.shape[0]
        log["reconstruction"] = self.decode_first_stage(z)

        # mask = torch.zeros(N, 1, 1, 1).cuda().float()
        # mask[[0, 1, 2]] = 1.0

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
            samples, z_denoise_row = self.sample_log(cond={"c_crossattn": [c_cross],
                                                           'pose': c['pose'],
                                                           'c_concat': c_concat,
                                                           'c_adm': c_adm,
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
            # from data.datasets.panorama_datasets import concat_ray_bundle
            # ray = concat_ray_bundle(ray, ray)
            uc_full = {"c_crossattn": [uc_cross], 'c_adm': torch.zeros_like(c_adm), 'c_concat': None,
                       'pose': c['pose'], 'intrinsic': c["intrinsic"], 'dist': c['dist']}
            samples_cfg, intermediates = self.sample_log(
                cond={"c_crossattn": [c_cross], 'c_adm': c_adm, 'c_concat': c_concat,
                      'pose': c['pose'], 'intrinsic': c["intrinsic"], 'dist': c['dist']},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                # mask=None, x0=z,
            )

            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            # log['view'] = self.decode_first_stage(intermediates['rgb'][-1])

        result_grid = make_grid(
            torch.cat([
                log["reconstruction"],
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"],
                # log["view"],
            ], dim=0), nrow=N)
        log = {'result': result_grid}
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        h, w = 32, 32
        shape = (self.channels, h // 1, w // 1)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        self.model.diffusion_model.input_blocks.requires_grad_(False)
        self.model.diffusion_model.time_embed.requires_grad_(False)
        self.model.diffusion_model.label_emb.requires_grad_(False)
        lr = self.learning_rate
        p1 = []
        p2 = []
        # params = list(self.control_model.parameters())
        # if not self.sd_locked:
        #     params += list(self.model.diffusion_model.parameters())
        for n, p in self.control_model.named_parameters():
            if 'fusion_module' in n:
                p1.append(p)
            else:
                p2.append(p)
        print('fusion module: ', len(p1))
        opt = torch.optim.AdamW([
            {"params": p1, "lr": lr},
            {"params": p2, "lr": lr}
        ], lr=lr)
        # opt = torch.optim.AdamW(params, lr=lr)
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


class ControlNet(nn.Module):
    def __init__(self,
                 hint_channels,
                 image_size,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 num_classes=None,
                 use_checkpoint=False,
                 use_fp16=False,
                 use_bf16=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 num_heads_upsample=-1,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 use_new_attention_order=False,
                 use_spatial_transformer=False,  # custom transformer support
                 transformer_depth=1,  # custom transformer support
                 adm_in_channels=None,  # custom transformer support
                 n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
                 legacy=True,
                 disable_self_attentions=None,
                 num_attention_blocks=None,
                 disable_middle_self_attn=False,
                 use_linear_in_transformer=False,
                 num_tri_layers=4,
                 context_dim=1024,
                 num_samples_per_ray=24):
        super().__init__()

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
        self.time_embed_dim = time_embed_dim
        self.dims = dims
        self.num_classes = num_classes
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        self.label_emb = nn.Sequential(
            nn.Sequential(
                linear(adm_in_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        )

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.control_embed = nn.Sequential(
            conv_nd(dims, hint_channels, model_channels, 3, padding=1),
        )

        self.gather_module = MultiFusionModule(model_channels, use_checkpoint=use_checkpoint,
                                               emb_channels=time_embed_dim,
                                               use_scale_shift_norm=use_scale_shift_norm)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        # self.fusion_module = nn.ModuleList([MultiFusionModule(model_channels, use_checkpoint=use_checkpoint)])
        # self.fusion_module = nn.ModuleList([
        #     nn.ModuleList([
        #         conv_nd(self.dims, model_channels, model_channels, 1, padding=0),
        #         zero_module(conv_nd(self.dims, model_channels, model_channels, 1, padding=0))
        #     ])
        # ])
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
                self.zero_convs.append(self.make_zero_conv(ch))
                # self.fusion_module.append(nn.Identity())
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
                self.zero_convs.append(self.make_zero_conv(ch))
                # self.fusion_module.append(
                #     nn.ModuleList([
                #         conv_nd(self.dims, ch, model_channels, 1, padding=0),
                #         zero_module(conv_nd(self.dims, model_channels, ch, 1, padding=0))
                #     ])
                # )
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
        # self.middle_fusion_module = nn.ModuleList([
        #     conv_nd(self.dims, ch, model_channels, 1, padding=0),
        #     zero_module(conv_nd(self.dims, model_channels, ch, 1, padding=0))
        # ])
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.dims = dims
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        self.embedding_dropout = 0.5

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def make_norm_conv(self, channels, out_channels):
        return TimestepEmbedSequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, channels, out_channels, 3, padding=1))
        )

    def forward(self, x_t, timesteps, context, ref_img, pose, intrinsic, dist, y):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb = emb + self.label_emb(y)

        outs = []
        guided_hint = True

        if self.training:
            mask = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(x_t.shape[0], device=x_t.device))
            mask = mask[:, None, None, None]
        else:
            if ref_img is None:
                mask = torch.bernoulli((torch.zeros(x_t.shape[0], device=x_t.device)))
                mask = mask[:, None, None, None]
            else:
                mask = torch.bernoulli((torch.ones(x_t.shape[0], device=x_t.device)))
                mask = mask[:, None, None, None]

        B, N = pose.size(0), pose.size(1)
        h = x_t.type(self.dtype)
        x = h.chunk(dim=0, chunks=N)
        if ref_img is None:
            ref_img = torch.zeros_like(x[0])
        mask = mask.chunk(dim=0, chunks=N)
        h = torch.cat([
            torch.cat([x[i], mask[i] * ref_img], dim=1).type(self.dtype) for i in range(N)
        ], dim=0)
        # else:
        #     h = torch.cat([
        #         torch.cat([x[i], ref_img], dim=1).type(self.dtype) for i in range(N)
        #     ], dim=0)
        emb_tmp = emb.chunk(dim=0, chunks=N)[0]
        for module, conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint:
                h = module(h, emb, context)
                guided_hint = False

                x1 = h.chunk(dim=0, chunks=N)
                x2 = self.gather_module(x_t, pose, intrinsic, dist, ref_img=ref_img, emb=emb_tmp, mask=mask)
                h = torch.cat([(x1[i] + mask[i] * x2[i]).type(self.dtype)
                               for i in range(N)], dim=0)
            else:
                h = module(h, emb, context)
            outs.append(conv(h, emb))

        h = self.middle_block(h, emb, context)
        # if ref_img is not None:
        #     x = h.chunk(dim=0, chunks=N)
        #     x = self.middle_fusion_module(x, pose, intrinsic, dist, mask=mask, ref_img=ref_img, emb=emb_tmp)
        #     h = torch.cat([x[i].type(self.dtype) for i in range(N)], dim=0)
        outs.append(self.middle_block_out(h, emb))
        return outs


class MultiFusionModule(nn.Module):
    def __init__(self, model_channels, use_checkpoint=False, dropout=0.0, emb_channels=0, use_scale_shift_norm=True):
        super().__init__()
        self.model_channels = model_channels
        self.hid_size = 320
        self.dims = 2
        self.render = Render(model_channels)
        resblock = partial(ResBlock, dropout=dropout, dims=self.dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=self.dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.conv_input = conv_nd(self.dims, 4, model_channels, 1, padding=0)
        self.conv_blocks_3d = nn.ModuleList([
            resblock3d(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
            resblock3d(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
            resblock3d(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
            resblock3d(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
        ])

        # self.attn_blocks = nn.ModuleList([
        #     TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
        #     TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
        #     TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
        #     TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
        # ])

        # self.conv_blocks = nn.ModuleList([
        #     resblock(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
        #     resblock(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
        #     resblock(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
        #     resblock(channels=model_channels, out_channels=model_channels, emb_channels=emb_channels),
        # ])
        self.mlp_out = conv_nd(self.dims + 1, model_channels, 16, 1, padding=0)

    def forward(self, x, pose, intrinsic, dist, emb=None, ref_img=None, mask=None):
        device = x.device
        B, N = pose.size(0), pose.size(1)
        H, W = x.size()[-2:]
        x = self.conv_input(x)
        ref_img = self.conv_input(ref_img)

        x = x.chunk(dim=0, chunks=N)
        aabb_scale = 1.0
        coord_grid = self.get_grid(1, size=(H, H, W), minval=0.0, maxval=1.0,
                                   device=device)
        # ref_img = self.ref_input(ref_img)
        all_img_coord = [[] for i in range(N)]
        for i in range(B):
            ori_focal, ori_principle_point = intrinsic[i].to(device).chunk(chunks=2, dim=1)
            focal, resolution = ori_focal, ori_principle_point * 2
            # build scenebox
            aabb = torch.tensor(
                [[-1 * aabb_scale, -1 * aabb_scale, -1 - float(dist[i])],
                 [1 * aabb_scale, 1 * aabb_scale, 1 - float(dist[i])]],
                dtype=torch.float32
            ).to(device)
            world_pos = coord_grid[0] * (aabb[1] - aabb[0]) + aabb[0]
            world_pos[..., 1] = -world_pos[..., 1]
            for j in range(N):
                pose_inv = torch.linalg.inv(pose[i, j]).to(device)
                cam_pos = torch.einsum('ij,zyxj->zyxi', pose_inv[:3, :3], world_pos) + pose_inv[:3, 3]
                cx = -cam_pos[..., 0] * focal[j] / cam_pos[..., 2]
                cy = -cam_pos[..., 1] * focal[j] / cam_pos[..., 2]

                cx = cx / ori_principle_point[j]
                cy = -cy / ori_principle_point[j]

                img_coord = torch.stack([cx, cy], dim=-1)  # [Z, Y, X, 2]
                all_img_coord[j].append(img_coord)

        vol = 0.0
        gt_vol = None
        cnt = 0
        for i in range(N):
            all_img_coord[i] = torch.stack(all_img_coord[i], dim=0)
            Z, Y, X, _ = all_img_coord[i].shape[-4:]
            sample_coord = all_img_coord[i].reshape(B, Z, Y * X, 2)
            img_vol = F.grid_sample(x[i], sample_coord, mode='bilinear', align_corners=True)
            img_vol = img_vol.reshape(B, -1, Z, Y, X)
            # print('fuck', mask[i].size())
            if i == 0:
                gt_vol = F.grid_sample(ref_img, sample_coord, mode='bilinear', align_corners=True)
                gt_vol = gt_vol.reshape(B, -1, Z, Y, X)
                break
            # vol += img_vol * mask[i][..., None]
            # cnt += mask[i][..., None]
        # vol = vol / N
        # vol = vol / (cnt.float() + 1e-10) + gt_vol

        for conv in self.conv_blocks_3d:
            gt_vol = conv(gt_vol, emb)

        # yx, zx, zy = vol.mean(dim=2), vol.mean(dim=3), vol.mean(dim=4)  # yx, zx, zy
        #
        # for i, (conv, attn) in enumerate(
        #         zip(self.conv_blocks, self.attn_blocks)):
        #     yx, zx, zy = conv(yx, emb), conv(zx, emb), conv(zy, emb)
        #     x_feats = attn(zx, yx, transpose_x1=True, transpose_x2=True)
        #     y_feats = attn(zy, yx, transpose_x1=True, transpose_x2=False)
        #     z_feats = attn(zy, zx, transpose_x1=False, transpose_x2=False)
        #     yx = yx + y_feats[1] + x_feats[1]
        #     zy = zy + y_feats[0] + z_feats[0]
        #     zx = zx + x_feats[0] + z_feats[1]
        # yx, zx, zy = torch.cat([yx, self.mlp_out(yx)], dim=1), \
        #              torch.cat([zx, self.mlp_out(zx)], dim=1), \
        #              torch.cat([zy, self.mlp_out(zy)], dim=1)
        # triplane = [yx, zx, zy]
        # feat = self.render(triplane, pose, intrinsic, dist)
        gt_vol = torch.cat([gt_vol, self.mlp_out(gt_vol)], dim=1)
        feat = self.render(gt_vol, pose, intrinsic, dist)

        return feat

    def get_grid(self, batchsize, size, minval=-1.0, maxval=1.0, device='cpu'):
        r"""Get a grid ranging [-1, 1] of 2D/3D coordinates.

        Args:
            batchsize (int) : Batch size.
            size (tuple) : (height, width) or (depth, height, width).
            minval (float) : minimum value in returned grid.
            maxval (float) : maximum value in returned grid.
        Returns:
            t_grid (4D tensor) : Grid of coordinates.
        """
        if not hasattr(self, 'coord_grid') or self.coord_grid.size(0) < batchsize:
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
            t_grid = t_grid.permute(0, 2, 3, 4, 1)
            setattr(self, 'coord_grid', t_grid.to(device))
        return self.coord_grid[:batchsize].to(device)


class Render(nn.Module):
    def __init__(self, model_channels, num_samples_per_ray=24):
        super().__init__()
        self.model_channels = model_channels
        self.num_samples_per_ray = num_samples_per_ray
        self.dims = 2

        self.sampler = SpacedSampler(
            spacing_fn=Identity(), spacing_fn_inv=Identity(),
            num_samples=num_samples_per_ray, single_jitter=False, train_stratified=True)
        self.direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        self.mlp_head = nn.Sequential(
            conv_nd(self.dims, self.direction_encoding.get_out_dim() + model_channels, model_channels, 1,
                    padding=0),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, model_channels, model_channels, 1, padding=0)),
        )

    @torch.no_grad()
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
        c_idx = torch.arange(0, nc)[:, None, None].expand(-1, img_coords.size(0), img_coords.size(1)).reshape(-1, 1)
        img_coords = img_coords[None].expand(nc, -1, -1, -1).reshape(-1, 2)
        ray_bundle = cameras.generate_rays(
            camera_indices=c_idx,
            coords=img_coords
        )
        return ray_bundle

    @torch.no_grad()
    def sample_points(self, x, pose, intrinsic, dist):
        pose = pose.type(x.dtype)
        intrinsic = intrinsic.type(x.dtype)
        B, _, _, H, W = x.size()
        aabb_scale = 1.0
        res1, res2 = [], []
        for i in range(B):
            ori_focal, ori_principle_point = intrinsic[i].chunk(chunks=2, dim=1)
            focal, resolution = ori_focal, ori_principle_point * 2

            cameras = Cameras(
                fx=focal,
                fy=focal,
                cx=resolution // 2,
                cy=resolution // 2,
                distortion_params=None,
                height=resolution.type(torch.IntTensor),
                width=resolution.type(torch.IntTensor),
                camera_to_worlds=pose[i][:, :3, :4],
                camera_type=CameraType.PERSPECTIVE,
            ).to(x.device)

            # build scenebox
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-1 * aabb_scale, -1 * aabb_scale, -1 - float(dist[i])],
                     [1 * aabb_scale, 1 * aabb_scale, 1 - float(dist[i])]],
                    dtype=torch.float32
                )
            )
            collider = AABBBoxCollider2(scene_box=scene_box, near_plane=0.01)

            # sample points
            ray_bundle = self.generate_rays(cameras, 256 // H)  # [N * H * W, 3]
            ray_bundle = collider(ray_bundle)
            ray_samples = self.sampler(ray_bundle)
            positions = ray_samples.frustums.get_positions()  # [N * H * W, Np, 3]
            positions = SceneBox.get_normalized_positions(positions, scene_box.aabb.to(x.device)).detach()
            positions = positions * 2 - 1
            positions[..., 1] = -positions[..., 1]
            # positions = positions.reshape(N * H * W, Np, 3)
            res1.append(positions)
            res2.append(ray_samples)
        return torch.stack(res1, dim=0), res2

    def forward(self, vol, pose, intrinsic, dist):
        B, N, Np = pose.size(0), pose.size(1), self.num_samples_per_ray
        H, W = vol.shape[-2:]

        def get_field(vol):
            color_field1, density_field1 = torch.split(vol, [self.model_channels, 16], dim=1)
            return color_field1, density_field1

        def get_density(field, coords):
            plane_features = F.grid_sample(
                field, coords, align_corners=True, mode='bilinear'
            )
            plane_features = torch.relu_(plane_features.sum(dim=1)).reshape(B, N * H * W, Np)[..., None]
            return plane_features

        def get_rgb(field, coords):
            d = torch.stack([rs.frustums.directions for rs in ray_samples], dim=0)
            d_encoded = self.direction_encoding(d).permute(0, 3, 1, 2)
            plane_features = F.grid_sample(
                field, coords, align_corners=True, mode='bilinear'
            ).reshape(B, -1, N * H * W, Np)
            plane_features = self.mlp_head(torch.cat([plane_features, d_encoded], dim=1))
            plane_features = plane_features.permute(0, 2, 3, 1)
            return plane_features

        # triplane: yx, zx, zy
        positions, ray_samples = self.sample_points(vol, pose, intrinsic, dist)
        # positions: B x (NxHxW) x Np x 3
        positions = positions.reshape(B, N, H * W, Np, 3)
        color_field, density_field = get_field(vol)

        density = get_density(density_field, positions)  # B x (NxHxW) x Np x 1
        feat = get_rgb(color_field, positions)  # B x (NxHxW) x Np x D
        # print(feat.size())
        weights = torch.stack([rs.get_weights(density[i]) for i, rs in enumerate(ray_samples)], dim=0)
        accumulated_weight = torch.sum(weights, dim=-2)  # B x (NxHxW) x Np x D
        comp_feat = torch.sum(weights * feat, dim=-2)
        comp_feat = comp_feat + (1.0 - accumulated_weight) * torch.zeros_like(feat[..., -1, :])
        # # B x (NxHxW) x D
        # comp_feat = feat.permute(0, 2, 3, 1).mean(dim=-2)
        comp_feat = comp_feat.reshape(B, N, H, W, -1).unbind(1)
        comp_feat = [i.permute(0, 3, 1, 2) for i in comp_feat]
        return comp_feat
        # B, N, Np = pose.size(0), pose.size(1), self.num_samples_per_ray
        # H, W = triplane[0].shape[-2:]
        #
        # def get_field(yx, zx, zy):
        #     color_field1, density_field1 = torch.split(yx, [self.model_channels, 16], dim=1)
        #     color_field2, density_field2 = torch.split(zx, [self.model_channels, 16], dim=1)
        #     color_field3, density_field3 = torch.split(zy, [self.model_channels, 16], dim=1)
        #     return [color_field1, color_field2, color_field3], [density_field1, density_field2, density_field3]
        #
        # def get_density(field, coords):
        #     plane_features = [
        #         F.grid_sample(
        #             field[0], coords[0], align_corners=True, mode='bilinear'
        #         ),
        #         F.grid_sample(
        #             field[1], coords[1], align_corners=True, mode='bilinear'
        #         ),
        #         F.grid_sample(
        #             field[2], coords[2], align_corners=True, mode='bilinear'
        #         )
        #     ]
        #     plane_features = torch.stack(plane_features, dim=1).sum(1)
        #     plane_features = torch.relu_(plane_features.sum(dim=1))[:, :, :, None]
        #     return plane_features
        #
        # def get_rgb(field, coords):
        #     d = torch.stack([rs.frustums.directions for rs in ray_samples], dim=0)
        #     d_encoded = self.direction_encoding(d).permute(0, 3, 1, 2)
        #     plane_features = [
        #         F.grid_sample(
        #             field[0], coords[0], align_corners=True, mode='bilinear'
        #         ),
        #         F.grid_sample(
        #             field[1], coords[1], align_corners=True, mode='bilinear'
        #         ),
        #         F.grid_sample(
        #             field[2], coords[2], align_corners=True, mode='bilinear'
        #         )
        #     ]
        #     plane_features = torch.stack(plane_features, dim=1).sum(1)
        #     plane_features = self.mlp_head(torch.cat([plane_features, d_encoded], dim=1))
        #     plane_features = plane_features.permute(0, 2, 3, 1)
        #     return plane_features
        #
        # # triplane: yx, zx, zy
        # positions, ray_samples = self.sample_points(triplane[0], pose, intrinsic, dist)
        # plane_coords = [positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]]
        # # positions: B x (NxHxW) x Np x 3
        # color_field, density_field = get_field(*triplane)
        #
        # density = get_density(density_field, plane_coords)  # B x (NxHxW) x Np x 1
        # feat = get_rgb(color_field, plane_coords)  # B x (NxHxW) x Np x D
        # # print(feat.size())
        # weights = torch.stack([rs.get_weights(density[i]) for i, rs in enumerate(ray_samples)], dim=0)
        # accumulated_weight = torch.sum(weights, dim=-2)  # B x (NxHxW) x Np x D
        # comp_feat = torch.sum(weights * feat, dim=-2)
        # comp_feat = comp_feat + (1.0 - accumulated_weight) * torch.zeros_like(feat[..., -1, :])
        # # # B x (NxHxW) x D
        # comp_feat = comp_feat.reshape(B, N, H, W, -1).unbind(1)
        # comp_feat = [i.permute(0, 3, 1, 2) for i in comp_feat]
        # return comp_feat


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


class Identity:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
