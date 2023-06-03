import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
from PIL import Image
import PIL

from diffusers import DDIMScheduler

import os
import sys
from os import path

from guidance.sd_unclip_utils import StableDiffusionUnclip
from guidance.sd_utils import StableDiffusion

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import cv2
from PIL import Image
import json

from ldm.util import instantiate_from_config


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'[INFO] Loaded state_dict from [{ckpt_path}]')
    return state_dict


def load_checkpoint(model, resume):
    resume = resume
    print('[INFO] Loading ckpt from {}...'.format(resume))
    sd = load_state_dict(resume, location='cpu')
    print(model.load_state_dict(sd, strict=False))


# load model
def load_model_from_config(config_file, ckpt, device, vram_O=False, verbose=False):
    # import pdb; pdb.set_trace()

    # create model
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model).cpu()
    print(f'[INFO] Loaded model config from [{config_file}]')
    load_checkpoint(model, ckpt)  # for debug
    model = model.to(device)
    model.sd_locked = True
    model.only_mid_control = False
    model.training = False

    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()
    model.eval().to(device)

    return model, config


class DreamScene(nn.Module):
    def __init__(self, device, fp16,
                 config="./pretrained/dreamscene/rldm_vit_l3.yaml",
                 ckpt="/workspace/ControlNet/scene/final_image_cond8/model-epoch=66-global_step=83499.0.ckpt",
                 vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt

        self.model, self.config = load_model_from_config(config, ckpt, device, self.vram_O)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # self.sd_model = StableDiffusionUnclip(device, fp16, False)
        self.sd_model2 = StableDiffusion(device, fp16, False)
        # del self.model.vae.decoder

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [B, 3, 256, 256] in [-1, 1]

        cs, vs, cadms = [], [], []  # c_crossattn, c_concat
        for xx in x:
            if len(xx.shape) == 3:
                xx = xx.unsqueeze(0)
            c, v, cadm = self.model.get_img_embeds(xx)
            cs.append(c)
            vs.append(v)
            cadms.append(cadm)
        return cs, vs, cadms

    def get_text_embeds(self, x):
        return self.sd_model2.get_text_embeds(x) #  .get_learned_conditioning(x)
        # inputs = self.sd_model.tokenizer(x, padding='max_length', max_length=self.tokenizer.model_max_length,
        #                                  return_tensors='pt')
        # embeddings = self.sd_model.text_encoder(inputs.input_ids.to(self.device))[0]
        # return embeddings
        # return self.model.get_text_embeds(x)

    def train_step(self, embeddings, pred_rgb, pose, intrinsic, dist,
                   guidance_scale=3.0, as_latent=False, grad_scale=1, save_guidance_path: Path = None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]
        # adjust SDS scale based on how far the novel view is from the known view
        loss1 = self.sd_model2.train_step(
            torch.cat([embeddings['neg_prompt_embeds'], embeddings['prompt_embeds']], dim=0),
            pred_rgb,
            guidance_scale,
        )
        return loss1
        # text_embeddings = torch.cat([embeddings['neg_prompt_embeds'], embeddings['prompt_embeds']], dim=0)

        n_pose = pose.size(1)
        if n_pose > 1:
            intrinsic = intrinsic.repeat(n_pose, 1)
            dist = dist.view(1).repeat(n_pose)
        pose = torch.cat(pose.unbind(dim=1), dim=0)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=True) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False) * 2 - 1
            latents_768 = self.encode_imgs(pred_rgb_256)

        # t = torch.randint(self.min_step, self.max_step + 1, (latents_768.shape[0],), dtype=torch.long,
        #                   device=self.device)

        # q sample
        with torch.no_grad():
            # noise = torch.randn_like(latents)

            t_in = torch.cat([t] * 2).to(self.device)

            noise_preds_sd = []
            num_imgs = len(embeddings["c_crossattn"])

            noise_768 = torch.randn_like(latents_768)
            latents_noisy_768 = self.scheduler.add_noise(latents_768, noise_768, t)
            x_in = torch.cat([latents_noisy_768] * 2)

            # model_output_sd = self.sd_model.unet(x_in, t_in, encoder_hidden_states=text_embeddings).sample

            for idx in range(num_imgs):

                c_crossattn, c_concat, c_adm = embeddings["c_crossattn"][idx], embeddings["c_concat"][idx], \
                                               embeddings["c_adm"][idx]
                cond = {"c_crossattn": [c_crossattn], "c_concat": [c_concat], "c_adm": c_adm, "pose": pose,
                        "intrinsic": intrinsic, "dist": dist.view(1)}
                uncond = {"c_crossattn": [self.model.get_unconditional_conditioning(1)],
                          "c_concat": [c_concat], 'c_adm': torch.zeros_like(c_adm),
                          "pose": pose, "intrinsic": intrinsic, "dist": dist.view(1)}
                c_in = dict()
                for k in cond:
                    if isinstance(cond[k], list):
                        c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
                    elif isinstance(cond[k], torch.Tensor):
                        c_in[k] = torch.cat([uncond[k], cond[k]])
                    else:
                        c_in[k] = cond[k]

            # import pdb; pdb.set_trace()
            model_output_sd, render_rgb = self.model.apply_model(x_in, t_in, c_in, return_rgb=True)
            model_uncond_sd, model_t_sd = model_output_sd.chunk(2)
            render_rgb = render_rgb.chunk(2)[1]
            model_output_sd = model_uncond_sd + guidance_scale * (model_t_sd - model_uncond_sd)
            e_t_sd = self.model.predict_eps_from_z_and_v(latents_noisy_768, t, model_output_sd)

            noise_preds_sd.append(e_t_sd)
        noise_pred_sd = torch.stack(noise_preds_sd).sum(dim=0) / len(noise_preds_sd)

        w = (1 - self.alphas[t])
        grad = (grad_scale * w)[:, None, None, None] * (noise_pred_sd - noise_768)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            pred_rgb_256 = torch.clamp((pred_rgb_256 + 1.0) / 2.0, min=0.0, max=1.0)
            result_hopefully_less_noisy_image2 = self.decode_latents(
                self.model.predict_start_from_noise(latents_noisy_768, t, noise_pred_sd))
            # visualize noisier image
            result_noisier_image = self.decode_latents(latents_noisy_768)
            rendered_imgs = self.decode_latents(render_rgb)
            # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
            # print(pred_rgb_256.size(), result_hopefully_less_noisy_image2.size(),
            #       result_noisier_image.size(), rendered_imgs.size())
            # exit(0)
            viz_images = torch.cat(
                [pred_rgb_256, result_noisier_image,
                 result_hopefully_less_noisy_image2, rendered_imgs
                 ], dim=-1)
            save_image(viz_images, save_guidance_path)

        loss = SpecifyGradient.apply(latents_768, grad)
        return loss1 + loss

    # def train_step(self, embeddings, pred_rgb, pose, intrinsic, dist,
    #                guidance_scale=3, as_latent=False, grad_scale=1, save_guidance_path: Path = None):
    #     # pred_rgb: tensor [1, 3, H, W] in [0, 1]
    #     # adjust SDS scale based on how far the novel view is from the known view
    #
    #     n_pose = pose.size(1)
    #     if n_pose > 1:
    #         intrinsic = intrinsic.repeat(n_pose, 1)
    #         dist = dist.view(1).repeat(n_pose)
    #     pose = torch.cat(pose.unbind(dim=1), dim=0)
    #
    #     if as_latent:
    #         latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=True) * 2 - 1
    #     else:
    #         pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=True) * 2 - 1
    #         latents = self.encode_imgs(pred_rgb_256)
    #
    #         pred_rgb_768 = F.interpolate(pred_rgb, (768, 768), mode="bilinear", align_corners=True) * 2 - 1
    #         latents_768 = self.encode_imgs(pred_rgb_768)
    #
    #     t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
    #
    #     # q sample
    #     with torch.no_grad():
    #         noise = torch.randn_like(latents)
    #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
    #         x_in = torch.cat([latents_noisy] * 2)
    #         t_in = torch.cat([t] * 2).to(self.device)
    #
    #         noise_preds = []
    #         noise_preds_sd = []
    #         num_imgs = len(embeddings["c_crossattn"])
    #         for idx in range(num_imgs):
    #
    #             c_crossattn, c_concat, c_adm = embeddings["c_crossattn"][idx], embeddings["c_concat"][idx], \
    #                                            embeddings["c_adm"][idx]
    #             cond = {"c_crossattn": [c_crossattn], "c_concat": [c_concat], "c_adm": c_adm, "pose": pose,
    #                     "intrinsic": intrinsic, "dist": dist.view(1)}
    #             uncond = {"c_crossattn": [self.model.get_unconditional_conditioning(1)],
    #                       "c_concat": [c_concat], 'c_adm': torch.zeros_like(c_adm),
    #                       "pose": pose, "intrinsic": intrinsic, "dist": dist.view(1)}
    #             c_in = dict()
    #             for k in cond:
    #                 if isinstance(cond[k], list):
    #                     c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
    #                 elif isinstance(cond[k], torch.Tensor):
    #                     c_in[k] = torch.cat([uncond[k], cond[k]])
    #                 else:
    #                     c_in[k] = cond[k]
    #
    #         # import pdb; pdb.set_trace()
    #         model_output, render_rgb = self.model.apply_model(x_in, t_in, c_in, return_rgb=True)
    #         model_uncond, model_t = model_output.chunk(2)
    #
    #         render_rgb = render_rgb.chunk(2)[1]
    #         model_output = model_uncond + guidance_scale * (model_t - model_uncond)
    #         if self.model.parameterization == "v":
    #             e_t = self.model.predict_eps_from_z_and_v(latents_noisy, t, model_output)
    #         else:
    #             e_t = model_output
    #
    #         noise_768 = torch.randn_like(latents_768)
    #         latents_noisy_768 = self.scheduler.add_noise(latents_768, noise_768, t)
    #         x_in = torch.cat([latents_noisy_768] * 2)
    #
    #         img_embeds = embeddings["c_adm"][0]
    #         text_embeds = embeddings['prompt_embeds']
    #         neg_text_embeds = embeddings['neg_prompt_embeds']
    #         cond = {"c_crossattn": [text_embeds], "c_adm": img_embeds}
    #         uncond = {"c_crossattn": [neg_text_embeds], 'c_adm': torch.zeros_like(img_embeds)}
    #         c_in = dict()
    #         for k in cond:
    #             if isinstance(cond[k], list):
    #                 c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
    #             elif isinstance(cond[k], torch.Tensor):
    #                 c_in[k] = torch.cat([uncond[k], cond[k]])
    #             else:
    #                 c_in[k] = cond[k]
    #
    #         model_output_sd = self.sd_model.model.apply_model(x_in, t_in, c_in)
    #         model_uncond_sd, model_t_sd = model_output_sd.chunk(2)
    #         model_output_sd = model_uncond_sd + guidance_scale * (model_t_sd - model_uncond_sd)
    #         if self.model.parameterization == "v":
    #             e_t_sd = self.model.predict_eps_from_z_and_v(latents_noisy_768, t, model_output_sd)
    #         else:
    #             e_t_sd = model_output_sd
    #
    #         noise_preds.append(e_t)
    #         noise_preds_sd.append(e_t_sd)
    #     noise_pred = torch.stack(noise_preds).sum(dim=0) / len(noise_preds)
    #     noise_pred_sd = torch.stack(noise_preds_sd).sum(dim=0) / len(noise_preds_sd)
    #
    #     w = (1 - self.alphas[t])
    #     # grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)
    #     # grad = (grad_scale * w)[:, None, None, None] * (noise_pred_sd - noise_pred)
    #     # grad = (grad_scale * w)[:, None, None, None] * (-noise_pred)
    #     # grad2 = (grad_scale * w)[:, None, None, None] * noise_pred_sd
    #     grad = (grad_scale * w)[:, None, None, None] * (noise_pred_sd - noise_768)
    #     grad = torch.nan_to_num(grad)
    #     # grad2 = torch.nan_to_num(grad2)
    #
    #     if save_guidance_path:
    #         with torch.no_grad():
    #             pred_rgb_256 = torch.clamp((pred_rgb_256 + 1.0) / 2.0, min=0.0, max=1.0)
    #
    #         # visualize predicted denoised image
    #         result_hopefully_less_noisy_image = self.decode_latents(
    #             self.model.predict_start_from_noise(latents_noisy, t, noise_pred))
    #         result_hopefully_less_noisy_image2 = self.decode_latents(
    #             self.model.predict_start_from_noise(latents_noisy_768, t, noise_pred_sd))
    #         result_hopefully_less_noisy_image2 = \
    #             F.interpolate(result_hopefully_less_noisy_image2, (256, 256), mode="bilinear", align_corners=True)
    #         # visualize noisier image
    #         result_noisier_image = self.decode_latents(latents_noisy)
    #
    #         rendered_imgs = self.decode_latents(render_rgb)
    #         # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
    #         viz_images = torch.cat(
    #             [pred_rgb_256, result_noisier_image, result_hopefully_less_noisy_image,
    #              result_hopefully_less_noisy_image2, rendered_imgs], dim=-1)
    #         save_image(viz_images, save_guidance_path)
    #
    #     # since we omitted an item in grad, we need to use the custom function to specify the gradient
    #     loss = SpecifyGradient.apply(latents_768, grad)  # + SpecifyGradient.apply(latents_768, grad2)
    #
    #     return loss  # + 2.0 * F.mse_loss(render_rgb, latents)

    def __call__(self, image, text,
                 scale=3, ddim_steps=50, ddim_eta=0.0, h=768, w=768,
                 c_crossattn=None, c_concat=None, c_adm=None, post_process=True
                 ):

        with torch.no_grad():
            if c_crossattn is None:
                if len(image[0].shape) == 3:
                    image = [img.unsqueeze(0) for img in image]

                embeddings = self.get_img_embeds(image)
                embeddings = {'c_crossattn': embeddings[0],
                              'c_concat': embeddings[1],
                              'c_adm': embeddings[2], }

                # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # produce latents loop
            latents = torch.randn((1, 4, h // 8, w // 8), device=self.device)
            self.scheduler.set_timesteps(20)
            recons = self.model.decode_first_stage(embeddings['c_concat'][0]).clamp(-1.0, 1.0)
            text_embeds = self.get_text_embeds(text)
            neg_text_embeds = self.get_text_embeds([""])
            img_embeds = embeddings["c_adm"][0]

            # img_embeds = torch.zeros_like(img_embeds)
            # text_embeds = torch.zeros_like(text_embeds)
            # neg_text_embeds = torch.zeros_like(neg_text_embeds)

            for i, t in enumerate(self.scheduler.timesteps):
                t_int = t
                t = torch.full((1,), t, device=self.device, dtype=torch.long)
                print(i, t)
                x_in = torch.cat([latents] * 2)
                t_in = torch.cat([t] * 2).to(self.device)

                assert not torch.isnan(x_in).any()
                assert not torch.isnan(t_in).any()

                cond = {"c_crossattn": [text_embeds], "c_adm": img_embeds}
                uncond = {"c_crossattn": [neg_text_embeds], 'c_adm': torch.zeros_like(img_embeds)}
                c_in = dict()
                for k in cond:
                    if isinstance(cond[k], list):
                        c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
                    elif isinstance(cond[k], torch.Tensor):
                        c_in[k] = torch.cat([uncond[k], cond[k]])
                    else:
                        c_in[k] = cond[k]

                model_output = self.sd_model.model.apply_model(x_in, t_in, c_in)
                model_uncond, model_t = model_output.chunk(2)
                model_output = model_uncond + scale * (model_t - model_uncond)
                if self.model.parameterization == "v":
                    e_t = self.sd_model.model.predict_eps_from_z_and_v(latents, t, model_output)
                else:
                    e_t = model_output
                assert not torch.isnan(model_output).any()
                # print('fuck', t.device, self.sd_model.model.alphas_cumprod.device, e_t.device, latents.device)
                latents = self.scheduler.step(e_t, t_int, latents, eta=ddim_eta)['prev_sample']

            imgs = self.decode_latents(latents)
            print(imgs.amax(), imgs.amin(), imgs.mean())
            imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs
            return imgs, recons.cpu().numpy().transpose(0, 2, 3, 1) if post_process else recons

    # verification
    # @torch.no_grad()
    # def __call__(self, image, pose, instrinsic, dist,
    #              scale=3, ddim_steps=50, ddim_eta=1, h=256, w=256,
    #              c_crossattn=None, c_concat=None, c_adm=None, post_process=True
    #              ):
    #
    #     if c_crossattn is None:
    #         if len(image[0].shape) == 3:
    #             image = [img.unsqueeze(0) for img in image]
    #
    #         embeddings = self.get_img_embeds(image)
    #         embeddings = {'c_crossattn': embeddings[0],
    #                       'c_concat': embeddings[1],
    #                       'c_adm': embeddings[2], }
    #
    #         # import pdb; pdb.set_trace()
    #         c_crossattn = embeddings["c_crossattn"]
    #     # import pdb; pdb.set_trace()
    #
    #     n_pose = pose.size(1)
    #     if n_pose > 1:
    #         instrinsic = instrinsic.repeat(n_pose, 1)
    #         dist = dist.view(1).repeat(n_pose)
    #     pose = torch.cat(pose.unbind(dim=1), dim=0)
    #
    #     if c_concat is None:
    #         c_concat = embeddings["c_concat"]
    #         recons = self.model.decode_first_stage(c_concat[0])
    #
    #     if c_adm is None:
    #         c_adm = embeddings["c_adm"]
    #
    #     cond = {"c_crossattn": c_crossattn, "c_concat": c_concat, "c_adm": c_adm[0], "pose": pose,
    #             "intrinsic": instrinsic, "dist": dist}
    #     for k, v in cond.items():
    #         if isinstance(v, (list, tuple)):
    #             print(k, len(v), v[0].shape)
    #         elif isinstance(v, torch.Tensor):
    #             print(k, v.shape)
    #         else:
    #             print(k, v)
    #     uncond = {"c_crossattn": [self.model.get_unconditional_conditioning(1)],
    #               "c_concat": c_concat,
    #               'c_adm': c_adm[0], "pose": pose, "intrinsic": instrinsic, "dist": dist.view(1)}
    #
    #     # produce latents loop
    #     latents = torch.randn((1, 4, h // 8, w // 8), device=self.device)
    #     self.scheduler.set_timesteps(ddim_steps)
    #
    #     for i, t in enumerate(self.scheduler.timesteps):
    #         x_in = torch.cat([latents] * 2)
    #         t_in = torch.cat([t.view(1)] * 2).to(self.device)
    #         c_in = dict()
    #         for k in cond:
    #             if isinstance(cond[k], list):
    #                 c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
    #             elif isinstance(cond[k], torch.Tensor):
    #                 c_in[k] = torch.cat([uncond[k], cond[k]])
    #             else:
    #                 c_in[k] = cond[k]
    #
    #         model_output, render_rgb = self.model.apply_model(x_in, t_in, c_in, return_rgb=True)
    #         model_uncond, model_t = model_output.chunk(2)
    #         render_rgb = render_rgb.chunk(2)[1]
    #         model_output = model_uncond + scale * (model_t - model_uncond)
    #         if self.model.parameterization == "v":
    #             e_t = self.model.predict_eps_from_z_and_v(latents, t.view(1).to(self.device), model_output)
    #         else:
    #             e_t = model_output
    #         latents = self.scheduler.step(e_t, t, latents, eta=ddim_eta)['prev_sample']
    #
    #     imgs = self.decode_latents(latents)
    #     imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs
    #     render_rgb = self.decode_latents(render_rgb)
    #     render_rgb = render_rgb.cpu().numpy().transpose(0, 2, 3, 1) if post_process else render_rgb
    #     return imgs, render_rgb, recons.cpu().numpy().transpose(0, 2, 3, 1) if post_process else recons

    def decode_latents(self, latents):

        imgs = self.model.decode_first_stage(latents)
        imgs = torch.clamp((imgs + 1.0) / 2.0, min=0.0, max=1.0)
        return imgs

    def encode_imgs(self, imgs):
        latents = torch.cat(
            [self.model.get_first_stage_encoding(self.model.encode_first_stage(img.unsqueeze(0))) for img in imgs],
            dim=0)
        return latents  # [B, 4, 32, 32] Latent space image


def get_numpy_image(image_filename, shape=None, scale_factor=1.0):
    """Returns the image of shape (H, W, 3 or 4).

    Args:
        image_idx: The image index in the dataset.
    """
    image = np.array(Image.open(image_filename))
    mask = image[:, :, 3]
    image[mask == 0] = 255
    pil_image = Image.fromarray(image[:, :, :3])
    if shape is not None:
        pil_image = pil_image.resize(shape, resample=Image.ANTIALIAS)
    elif scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.ANTIALIAS)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    return image


if __name__ == "__main__":
    import os
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import json

    #
    parser = argparse.ArgumentParser()
    # parser.add_argument('input', type=str, default="/mnt/cache_sail/views_release/4a03d2eceba847ea897f0944e8a57ab3/010.png")
    parser.add_argument('--fp16', action='store_true',
                        help="use float16 for training")  # no use now, can only run in fp32

    parser.add_argument('--posefile', type=str,
                        default="/mnt/cache_sail/liulj/stable-dreamfusion/ds_tests/posefile.json",
                        help="json file, consists of at least one conditional RT and one target RT, in world2cam format")

    opt = parser.parse_args()
    device = torch.device('cuda')
    #
    # print(f'[INFO] loading poses from {opt.posefile} ...')
    #
    # cams_file = json.load(open(opt.posefile, "r"))
    # # Nx4x4
    # target_RTs = []
    # for target in cams_file["targets"]:
    #     target_RT = target["target_rt"]
    #     target_RTs.append(torch.from_numpy(np.array(target_RT)))
    # target_RTs = torch.stack(target_RTs)
    # if "cond_rt" not in cams_file:
    #     # 4x4
    #     cond_RT = torch.from_numpy(target_RTs[0])
    # else:
    #     # 4x4
    #     cond_RT = torch.from_numpy(np.array(cams_file["cond_rt"]))
    #
    # relative_poses = torch.cat(
    #     [(cond_RT @ torch.linalg.inv(target_RTs[i])).unsqueeze(0) for i in range(target_RTs.shape[0])],
    #     dim=0).unsqueeze(0)
    # print(f"cond_RT: {cond_RT.shape}, \n{cond_RT}")
    # # print(f"target_RT: {target_RT.shape}")
    # # relative_poses = cond_RT @ torch.linalg.inv(target_RT)
    # # print(f"relative pose: \n{relative_poses}")
    # # relative_poses = relative_poses.unsqueeze(0)[:, :3, :4].unsqueeze(0)
    # print(f"relative pose: {relative_poses.shape}")
    #
    # source_dist = torch.norm(cond_RT[:3, 3], p=2)
    # intrinsic = torch.from_numpy(np.array([560 * 0.5, 256 * 0.5])).view(1, -1)
    #
    # cond_img_file = cams_file["cond_imgfile"]
    # print(f'[INFO] loading image from {cond_img_file} ...')

    image = get_numpy_image('./data/teddy.png', shape=(256, 256))
    image = (image.astype(np.float32) / 255.0) * 2 - 1
    image = torch.from_numpy(image).to(device)

    print(f"image: {image.shape}")
    # print(f"relative_poses: {relative_poses.shape}")
    # print(f"intrinsic: {intrinsic.shape}")

    # import pdb; pdb.set_trace()

    print(f'[INFO] loading model ...')
    model = DreamScene(device, opt.fp16)

    with torch.cuda.amp.autocast(enabled=opt.fp16):
        outputs = model([image], ["teddy bear"])
    images, recons = outputs
    os.makedirs("debug_res", exist_ok=True)
    for idx in range(images.shape[0]):
        # import pdb; pdb.set_trace()
        Image.fromarray((images[idx] * 255.0).astype(np.uint8)).save(f"/workspace/{idx}_rgb.png")
        # Image.fromarray((renders[idx] * 255.0).astype(np.uint8)).save(f"debug_res/{idx}_render.png")
        Image.fromarray((recons[idx] * 255.0).astype(np.uint8)).save(f"/workspace/{idx}_recon.png")
