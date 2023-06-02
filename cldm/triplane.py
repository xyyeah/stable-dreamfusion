from abc import abstractmethod
import math
from functools import partial

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer, SequenceTransformer
from ldm.util import exists

from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, ResBlock, TimestepEmbedSequential, AttentionBlock, \
    Downsample, Upsample

from nerfstudio.data.scene_box import SceneBox


class TriplaneGenerator(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 2), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.const = torch.nn.Parameter(torch.randn([2, hint_channels, 64, 64]))  # yz, xz
        self.input_xy_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_yz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_xz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        ch = model_channels
        resblock = partial(ResBlock, emb_channels=0, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.input_xy_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_yz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_xz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            ch = mult * model_channels

        #     if level != len(channel_mult) - 1:
        #         self.input_xy_block.append(resblock(channels=mult * model_channels,
        #                                             out_channels=mult * model_channels, down=True))
        #         self.input_yz_block.append(resblock(channels=mult * model_channels,
        #                                             out_channels=mult * model_channels, down=True))
        #         self.input_xz_block.append(resblock(channels=mult * model_channels,
        #                                             out_channels=mult * model_channels, down=True))
        #     input_block_chans = [ch]
        # for level, mult in enumerate(channel_mult[::-1]):
        self.input_xy_block.append(conv_nd(dims, ch, model_channels, 1))
        self.input_yz_block.append(conv_nd(dims, ch, model_channels, 1))
        self.input_xz_block.append(conv_nd(dims, ch, model_channels, 1))
        # self.mid_xy_block = conv_nd(dims, ch, model_channels, 1)
        # self.mid_yz_block = conv_nd(dims, ch, model_channels, 1)
        # self.mid_xz_block = conv_nd(dims, ch, model_channels, 1)

    def forward(self, x, emb):
        # yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
        #              F.pad(x, (32, 32, 32, 32), mode='constant', value=0)
        yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), x
        for i, (f_yz, f_xz, f_xy) in enumerate(zip(self.input_yz_block, self.input_xz_block, self.input_xy_block)):
            if isinstance(f_xz, TimestepBlock):
                yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
            else:
                yz, xz = f_yz(yz), f_xz(xz)
                xy = f_xy(xy)
            if i != len(self.input_xy_block) - 1:
                yz_next = yz + xz.mean(dim=2, keepdims=True) + xy.mean(dim=2)[..., None]
                xz_next = xz + yz.mean(dim=2, keepdims=True) + xy.mean(dim=3, keepdims=True)
                xy_next = xy + yz.mean(dim=3)[..., None, :] + xz.mean(dim=3, keepdims=True)
                yz, xz, xy = yz_next, xz_next, xy_next
        return xy, yz, xz


class TriplaneGenerator2(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 2), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.const = torch.nn.Parameter(torch.randn([2, hint_channels, 64, 64]))  # yz, xz
        self.input_xy_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_yz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_xz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        ch = model_channels
        resblock = partial(ResBlock, emb_channels=0, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.input_xy_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_yz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_xz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            ch = mult * model_channels
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_xy_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                self.input_yz_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                self.input_xz_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                input_block_chans.append(ch)
        self.mid_xy_block = conv_nd(dims, ch, ch, 1)
        self.mid_yz_block = conv_nd(dims, ch, ch, 1)
        self.mid_xz_block = conv_nd(dims, ch, ch, 1)

        self.output_block = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(2):
                ich = input_block_chans.pop()
                layers = [resblock(channels=ch + ich, out_channels=mult * model_channels)]
                # self.output_block.append(resblock(channels=ch + ich,
                #                                   out_channels=mult * model_channels))
                ch = model_channels * mult
                if level > 0 and i == 1:
                    layers.append(resblock(channels=mult * model_channels,
                                           out_channels=mult * model_channels, up=True))
                self.output_block.append(TimestepEmbedSequential(*layers))

    def forward(self, x, emb):
        def fuse(yz, xz, xy):
            yz_next = yz + xz.mean(dim=2, keepdims=True) + xy.mean(dim=2)[..., None]
            xz_next = xz + yz.mean(dim=2, keepdims=True) + xy.mean(dim=3, keepdims=True)
            xy_next = xy + yz.mean(dim=3)[..., None, :] + xz.mean(dim=3, keepdims=True)
            yz, xz, xy = yz_next, xz_next, xy_next
            return yz, xz, xy

        yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), x
        outs = []
        for i, (f_yz, f_xz, f_xy) in enumerate(zip(self.input_yz_block, self.input_xz_block, self.input_xy_block)):
            if isinstance(f_xz, TimestepBlock):
                yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
            else:
                yz, xz, xy = f_yz(yz), f_xz(xz), f_xy(xy)
            yz, xz, xy = fuse(yz, xz, xy)
            outs.append(torch.cat([yz, xz, xy], dim=0))
        yz, xz, xy = self.mid_yz_block(yz), self.mid_xz_block(xz), self.mid_xy_block(xy)
        for module in self.output_block:
            yz1, xz1, xy1 = outs.pop().chunk(dim=0, chunks=3)
            xyz = torch.cat([
                torch.cat([yz, yz1], dim=1),
                torch.cat([xz, xz1], dim=1),
                torch.cat([xy, xy1], dim=1),
            ], dim=0)
            xyz = module(xyz, emb)
            yz, xz, xy = xyz.chunk(dim=0, chunks=3)
            yz, xz, xy = fuse(yz, xz, xy)
        return xy, yz, xz


class TriplaneGenerator3(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 2), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.const = torch.nn.Parameter(torch.randn([2, hint_channels, 64, 64]))  # yz, xz
        self.input_xy_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_yz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        self.input_xz_block = nn.ModuleList([conv_nd(dims, hint_channels, model_channels, 1)])
        ch = model_channels
        resblock = partial(ResBlock, emb_channels=0, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.input_xy_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_yz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            self.input_xz_block.append(resblock(channels=ch, out_channels=mult * model_channels))
            ch = mult * model_channels
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_xy_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                self.input_yz_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                self.input_xz_block.append(resblock(channels=mult * model_channels,
                                                    out_channels=mult * model_channels, down=True))
                input_block_chans.append(ch)
        self.mid_xy_block = conv_nd(dims, ch, ch, 1)
        self.mid_yz_block = conv_nd(dims, ch, ch, 1)
        self.mid_xz_block = conv_nd(dims, ch, ch, 1)

        self.output_xy_block = nn.ModuleList([])
        self.output_yz_block = nn.ModuleList([])
        self.output_xz_block = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(2):
                ich = input_block_chans.pop()
                layer_xy = [resblock(channels=ch + ich, out_channels=mult * model_channels)]
                layer_xz = [resblock(channels=ch + ich, out_channels=mult * model_channels)]
                layer_yz = [resblock(channels=ch + ich, out_channels=mult * model_channels)]
                ch = model_channels * mult
                if level > 0 and i == 1:
                    layer_xy.append(resblock(channels=mult * model_channels,
                                             out_channels=mult * model_channels, up=True))
                    layer_yz.append(resblock(channels=mult * model_channels,
                                             out_channels=mult * model_channels, up=True))
                    layer_xz.append(resblock(channels=mult * model_channels,
                                             out_channels=mult * model_channels, up=True))
                self.output_xy_block.append(TimestepEmbedSequential(*layer_xy))
                self.output_yz_block.append(TimestepEmbedSequential(*layer_yz))
                self.output_xz_block.append(TimestepEmbedSequential(*layer_xz))

    def forward(self, x, emb):
        def fuse(yz, xz, xy):
            yz_next = yz + xz.mean(dim=2, keepdims=True) + xy.mean(dim=2)[..., None]
            xz_next = xz + yz.mean(dim=2, keepdims=True) + xy.mean(dim=3, keepdims=True)
            xy_next = xy + yz.mean(dim=3)[..., None, :] + xz.mean(dim=3, keepdims=True)
            yz, xz, xy = yz_next, xz_next, xy_next
            return yz, xz, xy

        yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), x
        outs = []
        for i, (f_yz, f_xz, f_xy) in enumerate(zip(self.input_yz_block, self.input_xz_block, self.input_xy_block)):
            if isinstance(f_xz, TimestepBlock):
                yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
            else:
                yz, xz, xy = f_yz(yz), f_xz(xz), f_xy(xy)
            yz, xz, xy = fuse(yz, xz, xy)
            outs.append(torch.cat([yz, xz, xy], dim=0))
        yz, xz, xy = self.mid_yz_block(yz), self.mid_xz_block(xz), self.mid_xy_block(xy)
        for module_xy, module_xz, module_yz in zip(self.output_xy_block, self.output_xz_block, self.output_yz_block):
            yz1, xz1, xy1 = outs.pop().chunk(dim=0, chunks=3)
            yz, xz, xy = module_yz(torch.cat([yz, yz1], dim=1), emb), \
                         module_xz(torch.cat([xz, xz1], dim=1), emb), module_xy(torch.cat([xy, xy1], dim=1), emb)
            yz, xz, xy = fuse(yz, xz, xy)
        return xy, yz, xz


class TriplaneGenerator4(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 2), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        size = 96
        self.const = torch.nn.Parameter(torch.randn([2, model_channels, size, size]))  # yz, xz

        self.map_xy = conv_nd(dims, hint_channels, model_channels, 3, padding=1)
        ch = model_channels
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        attnblock = partial(SequenceTransformer, d_head=64, depth=1, disable_self_attn=False,
                            use_checkpoint=use_checkpoint)
        self.map_yz = resblock(channels=model_channels, out_channels=model_channels, emb_channels=model_channels)
        self.map_xz = resblock(channels=model_channels, out_channels=model_channels, emb_channels=model_channels)

        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.input_xy_block.extend([
                attnblock(in_channels=ch, n_heads=ch // 64),
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
            ])
            self.input_yz_block.extend([
                attnblock(in_channels=ch, n_heads=ch // 64),
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0)
            ])
            self.input_xz_block.extend([
                attnblock(in_channels=ch, n_heads=ch // 64),
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0)
            ])
            ch = mult * model_channels

        self.input_xy_block.append(conv_nd(dims, ch, model_channels + 16, 1))
        self.input_yz_block.append(conv_nd(dims, ch, model_channels + 16, 1))
        self.input_xz_block.append(conv_nd(dims, ch, model_channels + 16, 1))

    def forward(self, x, emb):
        pad = 16
        yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
                     self.map_xy(F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0))
        # yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
        #              self.map_xy(x)
        bsz = xy.size(0)

        xy_vec = xy.mean(dim=(-1, -2))
        yz = self.map_yz(yz, xy_vec)
        xz = self.map_xz(xz, xy_vec)

        def reshape(x1, dim1):
            b, c, h, w = x1.shape
            if dim1 == 0:
                x1 = x1.permute(0, 2, 1, 3).reshape(b * h, c, w)
                return x1
            x1 = x1.permute(0, 3, 1, 2).reshape(b * w, c, h)
            return x1

        def reshape_inv(x1, dim1):
            b, c, l = x1.shape
            if dim1 == 0:
                x1 = x1.reshape(bsz, b // bsz, c, l).permute(0, 2, 1, 3)
            else:
                x1 = x1.reshape(bsz, b // bsz, c, l).permute(0, 2, 3, 1)
            return x1

        for i, (f_yz, f_xz, f_xy) in enumerate(zip(self.input_yz_block, self.input_xz_block, self.input_xy_block)):
            if isinstance(f_xz, TimestepBlock):
                yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
            elif isinstance(f_xz, SequenceTransformer):
                y_feats = reshape(xy, 1), reshape(yz, 0)
                x_feats = reshape(xy, 0), reshape(xz, 0)
                z_feats = reshape(yz, 1), reshape(xz, 1)
                # print(f_xz(torch.cat(y_feats, dim=-1)).size())
                y_feats = f_xz(torch.cat(y_feats, dim=-1)).chunk(dim=-1, chunks=2)
                x_feats = f_yz(torch.cat(x_feats, dim=-1)).chunk(dim=-1, chunks=2)
                z_feats = f_xy(torch.cat(z_feats, dim=-1)).chunk(dim=-1, chunks=2)
                y_feats = [reshape_inv(y_feats[0], 1), reshape_inv(y_feats[1], 0)]
                x_feats = [reshape_inv(x_feats[0], 0), reshape_inv(x_feats[1], 0)]
                z_feats = [reshape_inv(z_feats[0], 1), reshape_inv(z_feats[1], 1)]
                xy = xy + y_feats[0] + x_feats[0]
                yz = yz + y_feats[1] + z_feats[0]
                xz = xz + x_feats[1] + z_feats[1]
            else:
                yz, xz, xy = f_yz(yz), f_xz(xz), f_xy(xy)
            # if i != len(self.input_xy_block) - 1:
            #     yz_next = yz + xz.mean(dim=2, keepdims=True) + xy.mean(dim=2)[..., None]
            #     xz_next = xz + yz.mean(dim=2, keepdims=True) + xy.mean(dim=3, keepdims=True)
            #     xy_next = xy + yz.mean(dim=3)[..., None, :] + xz.mean(dim=3, keepdims=True)
            #     yz, xz, xy = yz_next, xz_next, xy_next
        return xy, yz, xz


# class TriplaneAttention(nn.Module):
#     def __init__(self, in_channels, n_heads, use_checkpoint=False):
#         super().__init__()
#         self.attn = SequenceTransformer(in_channels=in_channels, n_heads=n_heads, d_head=64, depth=1,
#                                         disable_self_attn=False, use_checkpoint=use_checkpoint)
#
#     def forward(self, x1, x2, transpose_x1=False, transpose_x2=False, down=1):
#         if transpose_x1:
#             x1 = x1.permute(0, 1, 3, 2)
#         if transpose_x2:
#             x2 = x2.permute(0, 1, 3, 2)
#         b, d, h1, w1 = x1.shape
#         b, _, h2, w2 = x2.shape
#         assert x1.shape[1] == d
#         assert h1 == h2
#         if down > 1 and h1 == h2:
#             x1 = F.interpolate(x1, (h1 // down, w1 // down), mode='bilinear')
#             x2 = F.interpolate(x1, (h2 // down, w2 // down), mode='bilinear')
#             x1, x2 = x1.permute(0, 2, 1, 3).reshape(b * h1 // down, d, w1 // down), \
#                      x2.permute(0, 2, 1, 3).reshape(b * h2 // down, d, w2 // down)
#         elif h1 < h2:
#             x1 = F.interpolate(x1, (h2, w1), mode='bilinear')
#             assert x1.size(2) == h2
#             x1, x2 = x1.permute(0, 2, 1, 3).reshape(b * h2, d, w1), x2.permute(0, 2, 1, 3).reshape(b * h2, d, w2)
#         x_in = torch.cat([x1, x2], dim=2)
#         if down > 1 and h1 == h2:
#             x1_out, x2_out = self.attn(x_in).split([w1 // down, w2 // down], dim=2)
#             x1, x2 = x1_out.reshape(b, h1 // down, d, w1 // down).permute(0, 2, 1, 3), \
#                      x2_out.reshape(b, h2 // down, d, w2 // down).permute(0, 2, 1, 3)
#             x1 = F.interpolate(x1, (h1, w1), mode='bilinear')
#             x2 = F.interpolate(x2, (h2, w2), mode='bilinear')
#         elif h1 < h2:
#             x1_out, x2_out = self.attn(x_in).split([w1, w2], dim=2)
#             x1, x2 = x1_out.reshape(b, h2, d, w1).permute(0, 2, 1, 3), x2_out.reshape(b, h2, d, w2).permute(0, 2, 1, 3)
#             x1 = F.interpolate(x1, (h1, w1), mode='bilinear')
#             assert x1.size(2) == h1
#         if transpose_x1:
#             x1 = x1.permute(0, 1, 3, 2)
#         if transpose_x2:
#             x2 = x2.permute(0, 1, 3, 2)
#         return x1, x2

class TriplaneAttention(nn.Module):
    def __init__(self, in_channels, n_heads, use_checkpoint=False):
        super().__init__()
        self.attn = SequenceTransformer(in_channels=in_channels, n_heads=n_heads, d_head=64, depth=1,
                                        disable_self_attn=False, use_checkpoint=use_checkpoint)

    def forward(self, x1, x2, transpose_x1=False, transpose_x2=False):
        if transpose_x1:
            x1 = x1.permute(0, 1, 3, 2)
        if transpose_x2:
            x2 = x2.permute(0, 1, 3, 2)
        b, d, h1, w1 = x1.shape
        b, _, h2, w2 = x2.shape
        assert x1.shape[1] == d
        assert h1 == h2
        x1, x2 = x1.permute(0, 2, 1, 3).reshape(b * h1, d, w1), \
                 x2.permute(0, 2, 1, 3).reshape(b * h2, d, w2)
        x_in = torch.cat([x1, x2], dim=2)
        x1_out, x2_out = self.attn(x_in).split([w1, w2], dim=2)
        x1, x2 = x1_out.reshape(b, h1, d, w1).permute(0, 2, 1, 3), \
                 x2_out.reshape(b, h2, d, w2).permute(0, 2, 1, 3)
        if transpose_x1:
            x1 = x1.permute(0, 1, 3, 2)
        if transpose_x2:
            x2 = x2.permute(0, 1, 3, 2)
        return x1, x2


class TriplaneGenerator5(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        # size = 96
        self.const = torch.nn.Parameter(torch.randn([2, model_channels, 3, 3]))  # yz, xz
        self.map_xy = nn.Sequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 128, model_channels, 3, padding=1),
        )

        self.global_pool = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            conv_nd(dims, model_channels, model_channels, 1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        ch = model_channels
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.map_yz = resblock(channels=model_channels, out_channels=model_channels, emb_channels=model_channels)
        self.map_xz = resblock(channels=model_channels, out_channels=model_channels, emb_channels=model_channels)

        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
            ]))
            self.input_yz_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            ]))
            self.input_xz_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            ]))
            ch = mult * model_channels

        self.output_xy_block = conv_nd(dims, ch, model_channels + 16, 1)
        self.output_xz_block = conv_nd(dims, ch, model_channels + 16, 1)
        self.output_yz_block = conv_nd(dims, ch, model_channels + 16, 1)

    def forward(self, x, emb):
        pad = 16
        yz, xz, xy = self.const[1].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
                     self.map_xy(F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0))
        # yz, xz, xy = self.const[0].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
        #              self.map_xy(x)
        bsz = xy.size(0)

        xy_vec = self.global_pool(xy)
        yz = self.map_yz(yz, xy_vec)
        xz = self.map_xz(xz, xy_vec)
        # 3, 6, 12, 24, 48, 96
        for i, (f_yz, f_xz, f_xy, attn) in enumerate(
                zip(self.input_yz_block, self.input_xz_block, self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False)
            y_feats = attn[1](yz, xy, transpose_x1=False, transpose_x2=True)
            z_feats = attn[2](yz, xz, transpose_x1=True, transpose_x2=True)
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
            # print(yz.size(), xz.size(), xy.size())
            # exit(0)
        return self.output_xy_block(xy), self.output_yz_block(yz), self.output_xz_block(xz)


class TriplaneGenerator6(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        size = 96
        self.const = torch.nn.Parameter(torch.randn([2, hint_channels, size, size]))  # yz, xz

        ch = model_channels
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.map_yz = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 64, 3, padding=1),
            resblock(channels=64, out_channels=128, emb_channels=model_channels),
            resblock(channels=128, out_channels=model_channels, emb_channels=model_channels),
        )
        self.map_xz = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 64, 3, padding=1),
            resblock(channels=64, out_channels=128, emb_channels=model_channels),
            resblock(channels=128, out_channels=model_channels, emb_channels=model_channels),
        )

        self.map_xy = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 64, 3, padding=1),
            resblock(channels=64, out_channels=128, emb_channels=0),
            resblock(channels=128, out_channels=model_channels, emb_channels=0),
        )

        # self.map_xy = nn.Sequential(
        #     conv_nd(dims, hint_channels, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 32, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 64, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 64, 128, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 128, model_channels, 3, padding=1),
        # )

        self.global_pool = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            conv_nd(dims, model_channels, model_channels, 1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
            ]))
            self.input_yz_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                # resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            ]))
            self.input_xz_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                # resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            ]))
            ch = mult * model_channels

        self.output_xy_block = conv_nd(dims, ch, model_channels + 16, 1)
        self.output_xz_block = conv_nd(dims, ch, model_channels + 16, 1)
        self.output_yz_block = conv_nd(dims, ch, model_channels + 16, 1)

    def forward(self, x, emb):
        pad = 16
        yz, xz, xy = self.const[1].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
                     self.map_xy(F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0), emb)
        bsz = xy.size(0)
        xy_vec = self.global_pool(xy)
        yz = self.map_yz(yz, xy_vec)
        xz = self.map_xz(xz, xy_vec)
        # 3, 6, 12, 24, 48, 96
        downs = [8, 4, 2, 1]
        for i, (f_yz, f_xz, f_xy, attn) in enumerate(
                zip(self.input_yz_block, self.input_xz_block, self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[1](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[2](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_yz(yz, emb), f_xz(xz, emb), f_xy(xy, emb)
        return self.output_xy_block(xy), self.output_yz_block(yz), self.output_xz_block(xz)


class TriplaneGenerator7(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        size = 96
        self.const = torch.nn.Parameter(torch.randn([2, hint_channels, size, size]))  # yz, xz, xy

        ch = model_channels
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.map_yz_xz = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 64, 3, padding=1),
            resblock(channels=64, out_channels=128, emb_channels=model_channels),
            resblock(channels=128, out_channels=model_channels, emb_channels=model_channels),
        )

        self.map_xy = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 64, 3, padding=1),
            resblock(channels=64, out_channels=128, emb_channels=0),
            resblock(channels=128, out_channels=model_channels, emb_channels=0),
        )

        self.global_pool = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            conv_nd(dims, model_channels, model_channels, 1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0),
            ]))
            # self.input_yz_block.append(TimestepEmbedSequential(*[
            #     resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
            #     # resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            # ]))
            # self.input_xz_block.append(TimestepEmbedSequential(*[
            #     resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
            #     # resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0, up=True)
            # ]))
            ch = mult * model_channels

        self.output_xy_block = conv_nd(dims, ch, model_channels + 16, 1)

    def forward(self, x, emb):
        pad = 16
        x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)
        yz, xz, xy = self.const[1].expand(x.size(0), -1, -1, -1), self.const[1].expand(x.size(0), -1, -1, -1), \
                     self.map_xy(x, emb)
        xy_vec = self.global_pool(xy)
        yz = self.map_yz_xz(yz + x, xy_vec)
        xz = self.map_yz_xz(xz + x, xy_vec)
        # 3, 6, 12, 24, 48, 96
        downs = [8, 4, 2, 1]
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[1](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[2](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(yz, emb), f_xy(xz, emb), f_xy(xy, emb)
        return self.output_xy_block(xy), self.output_xy_block(yz), self.output_xy_block(xz)


class TriplaneGenerator8(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.size = 96
        # self.const = torch.nn.Parameter(torch.randn([3, hint_channels, self.size, self.size]))  # yz, xz, xy
        self.to3d = conv_nd(dims, hint_channels, self.size * 32, 1, padding=0)
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.res3d_block = TimestepEmbedSequential(
            resblock3d(channels=32, out_channels=32, emb_channels=0),
            resblock3d(channels=32, out_channels=64, emb_channels=0),
            resblock3d(channels=64, out_channels=64, emb_channels=0),
            resblock3d(channels=64, out_channels=model_channels, emb_channels=0),
        )
        ch = model_channels
        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0),
            ]))
            ch = mult * model_channels

        self.output_xy_block = conv_nd(dims, ch, model_channels + 16, 1)

    def forward(self, x, emb):
        pad = 16
        x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)  # xy
        b, _, lx, ly = x.shape
        x3d = self.to3d(x).reshape(b, -1, self.size, lx, ly).permute(0, 1, 3, 4, 2)
        x3d = self.res3d_block(x3d, emb)
        xy, xz, yz = x3d.mean(dim=4), x3d.mean(dim=3), x3d.mean(dim=2)
        # 3, 6, 12, 24, 48, 96

        downs = [8, 4, 2, 1]
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[1](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[2](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(yz, emb), f_xy(xz, emb), f_xy(xy, emb)
        return self.output_xy_block(xy), self.output_xy_block(yz), self.output_xy_block(xz)


class TriplaneGenerator9(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, dropout=0.0,
                 channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.size = 96
        # self.const = torch.nn.Parameter(torch.randn([3, hint_channels, self.size, self.size]))  # yz, xz, xy
        self.to3d = conv_nd(dims, hint_channels, self.size * 32, 1, padding=0)
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.res3d_block = TimestepEmbedSequential(
            resblock3d(channels=32, out_channels=32, emb_channels=0),
            resblock3d(channels=32, out_channels=64, emb_channels=0),
            resblock3d(channels=64, out_channels=model_channels, emb_channels=0),
            # resblock3d(channels=64, out_channels=model_channels, emb_channels=0),
        )
        ch = model_channels
        self.input_xy_block = nn.ModuleList([])
        self.input_yz_block = nn.ModuleList([])
        self.input_xz_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels, emb_channels=0),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels, emb_channels=0),
            ]))
            ch = mult * model_channels

        self.output_xy_block = conv_nd(dims, ch, model_channels + 16, 1)

    def forward(self, x, emb, x_t=None, emb_t=None, x2x_t=None):
        pad = 16
        x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)  # xy
        b, _, lx, ly = x.shape
        x3d = self.to3d(x).reshape(b, -1, self.size, lx, ly).permute(0, 1, 3, 4, 2)
        x3d = self.res3d_block(x3d, emb)
        xy, xz, yz = x3d.mean(dim=4), x3d.mean(dim=3), x3d.mean(dim=2)
        # 3, 6, 12, 24, 48, 96

        downs = [8, 4, 2, 1]
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[1](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[2](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(torch.cat([yz, xz, xy], dim=0), emb).chunk(dim=0, chunks=3)
        return self.output_xy_block(torch.cat([yz, xz, xy], dim=0))  # .chunk(dim=0, chunks=3)


class TriplaneGenerator10(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.size = 64
        aabb_scale = 1.0
        self.whole_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.25 * aabb_scale, -1.25 * aabb_scale, -1.25 * aabb_scale],
                 [1.25 * aabb_scale, 1.25 * aabb_scale, 0.25 * aabb_scale]],
                dtype=torch.float32
            )
        )
        self.cur_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.00 * aabb_scale, -1.00 * aabb_scale, -1.00 * aabb_scale],
                 [1.00 * aabb_scale, 1.00 * aabb_scale, 0.00 * aabb_scale]],
                dtype=torch.float32
            )
        )
        # self.const = torch.nn.Parameter(torch.randn([3, hint_channels, self.size, self.size]))  # yz, xz, xy
        self.to3d = conv_nd(dims, hint_channels, self.size * 32, 1, padding=0)
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        self.res3d_block = TimestepEmbedSequential(
            resblock3d(channels=32, out_channels=32, emb_channels=emb_channels),
            resblock3d(channels=32, out_channels=32, emb_channels=emb_channels),
            resblock3d(channels=32, out_channels=64, emb_channels=emb_channels),
            resblock3d(channels=64, out_channels=64, emb_channels=emb_channels),
            # resblock3d(channels=64, out_channels=64, emb_channels=emb_channels),
            conv_nd(dims + 1, 64, model_channels // 2, 1),
        )
        self.before_ch = model_channels // 2
        ch = model_channels
        self.input_xy_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                # TriplaneAttention(in_channels=ch, n_heads=ch // 64),
                # TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
            ]))
            ch = mult * model_channels
        output_channels = model_channels + 16
        self.output_xy_block = conv_nd(dims, ch, output_channels, 1)

    def forward(self, x, emb, x_t, emb_t, x2x_t):
        pad = 16
        # x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)  # xy
        x = torch.cat([x.permute(0, 1, 3, 2), x_t.permute(0, 1, 3, 2)], dim=0)
        # emb = torch.cat([emb, emb_t], dim=0)
        b, _, lx, ly = x.shape
        x3d = self.to3d(x).reshape(b, -1, self.size, lx, ly).permute(0, 1, 3, 4, 2)
        x3d = self.res3d_block(x3d, torch.cat([emb, emb_t], dim=0))
        xy, xz, yz = x3d.mean(dim=4), x3d.mean(dim=3), x3d.mean(dim=2)

        # 3, 6, 12, 24, 48, 96

        xy, xy_t = xy.chunk(dim=0, chunks=2)
        xz, xz_t = xz.chunk(dim=0, chunks=2)
        yz, yz_t = yz.chunk(dim=0, chunks=2)

        expand_size = int(self.size * 1.5)

        def denormalize(x, aabb):
            aabb_lengths = aabb[1] - aabb[0]  # [3]
            return x * aabb_lengths + aabb[0]

        vol_t = xy_t[..., None] + yz_t[..., None, :, :] + xz_t[..., :, None, :]
        coord_grid = self.get_grid(vol_t.size(0), (expand_size, expand_size, expand_size), minval=0.0, maxval=1.0,
                                   device=x.device)
        coord_grid[..., 1] = 1.0 - coord_grid[..., 1]
        coord_grid = denormalize(coord_grid, self.whole_scene_box.aabb.to(x.device))
        coord_grid = SceneBox.get_normalized_positions(torch.einsum('bij,bxyzj->bxyzi', x2x_t[:, :3, :3], coord_grid),
                                                       self.cur_scene_box.aabb.to(x.device))
        coord_grid = 2 * coord_grid - 1
        coord_grid[..., 1] = -coord_grid[..., 1]  # N,es,es,es,3
        vol_feats = F.grid_sample(vol_t, coord_grid, align_corners=True, mode='bilinear')

        xy = torch.cat([F.pad(xy, (pad, pad, pad, pad), mode='constant', value=0), vol_feats.mean(dim=4)], dim=1)
        xz = torch.cat([F.pad(xz, (pad, pad, pad, pad), mode='constant', value=0), vol_feats.mean(dim=3)], dim=1)
        yz = torch.cat([F.pad(yz, (pad, pad, pad, pad), mode='constant', value=0), vol_feats.mean(dim=2)], dim=1)

        downs = [8, 4, 2, 1]
        emb = torch.cat([emb, emb, emb], dim=0)
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[0](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[0](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(torch.cat([yz, xz, xy], dim=0), emb).chunk(dim=0, chunks=3)
        return self.output_xy_block(torch.cat([yz, xz, xy], dim=0))

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
        if not hasattr(self, 'grid') or self.grid.size(0) < batchsize:
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
            setattr(self, 'grid', t_grid.to(device))
        return self.grid[:batchsize]


class TriplaneGenerator11(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.size = 64
        self.before_ch = model_channels // 2
        aabb_scale = 1.0
        self.whole_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.25 * aabb_scale, -1.25 * aabb_scale, -1.25 * aabb_scale],
                 [1.25 * aabb_scale, 1.25 * aabb_scale, 0.25 * aabb_scale]],
                dtype=torch.float32
            )
        )
        self.cur_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.00 * aabb_scale, -1.00 * aabb_scale, -1.00 * aabb_scale],
                 [1.00 * aabb_scale, 1.00 * aabb_scale, 0.00 * aabb_scale]],
                dtype=torch.float32
            )
        )
        # self.const = torch.nn.Parameter(torch.randn([3, hint_channels, self.size, self.size]))  # yz, xz, xy
        self.to3d = conv_nd(dims, hint_channels, self.size * 32, 1, padding=0)
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        self.input_3d_block = nn.ModuleList([])
        model_channels3d = 32
        ch = model_channels3d
        input_block_chans = [model_channels3d]
        for level, mult in enumerate(channel_mult):
            self.input_3d_block.append(resblock3d(channels=ch, out_channels=mult * model_channels3d,
                                                  emb_channels=emb_channels))
            ch = mult * model_channels3d
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_3d_block.append(resblock3d(channels=mult * model_channels3d,
                                                      out_channels=mult * model_channels3d, emb_channels=emb_channels,
                                                      down=True))

                input_block_chans.append(ch)
        self.mid_3d_block = conv_nd(dims + 1, ch, ch, 1)

        self.output_3d_block = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(2):
                ich = input_block_chans.pop()
                layer_xy = [
                    resblock3d(channels=ch + ich, out_channels=mult * model_channels3d, emb_channels=emb_channels)]
                ch = model_channels3d * mult
                if level > 0 and i == 1:
                    layer_xy.append(resblock3d(channels=mult * model_channels3d, emb_channels=emb_channels,
                                               out_channels=mult * model_channels3d, up=True))
                self.output_3d_block.append(TimestepEmbedSequential(*layer_xy))
        self.final_3d_block = conv_nd(dims + 1, ch, self.before_ch, 1)

        # fuse & triplane
        ch = model_channels
        self.input_xy_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
            ]))
            ch = mult * model_channels
        output_channels = model_channels + 16
        self.output_xy_block = conv_nd(dims, ch, output_channels, 1)

    def forward(self, x, emb, x_t, emb_t, x2x_t, pad=16):
        def denormalize(x, aabb):
            aabb_lengths = aabb[1] - aabb[0]  # [3]
            return x * aabb_lengths + aabb[0]

        # 3d denoising
        # x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)  # xy
        x = torch.cat([x.permute(0, 1, 3, 2), x_t.permute(0, 1, 3, 2)], dim=0)
        # emb = torch.cat([emb, emb_t], dim=0)
        b, _, lx, ly = x.shape
        x3d = self.to3d(x).reshape(b, -1, self.size, lx, ly).permute(0, 1, 3, 4, 2)
        # x3d = self.input_res3d_block(x3d, torch.cat([emb, emb_t], dim=0))

        outs = [x3d]
        emb3d = torch.cat([emb, emb_t], dim=0)
        # print(len(self.input_3d_block), len(self.output_3d_block))
        # exit(0)
        for i, module in enumerate(self.input_3d_block):
            x3d = module(x3d, emb3d)
            outs.append(x3d)
        x3d = self.mid_3d_block(x3d)
        for i, module in enumerate(self.output_3d_block):
            x3d = module(torch.cat([x3d, outs.pop()], dim=1), emb3d)
        x3d = self.final_3d_block(x3d)

        # 2d triplane fusion
        expand_size = int(self.size * 1.5)
        vol, vol_t = x3d.chunk(dim=0, chunks=2)

        coord_grid = self.get_grid(vol_t.size(0), (expand_size, expand_size, expand_size), minval=0.0, maxval=1.0,
                                   device=x.device)
        coord_grid[..., 1] = 1.0 - coord_grid[..., 1]
        coord_grid = denormalize(coord_grid, self.whole_scene_box.aabb.to(x.device))
        coord_grid = SceneBox.get_normalized_positions(torch.einsum('bij,bxyzj->bxyzi', x2x_t[:, :3, :3], coord_grid),
                                                       self.cur_scene_box.aabb.to(x.device))
        coord_grid = 2 * coord_grid - 1
        coord_grid[..., 1] = -coord_grid[..., 1]  # N,es,es,es,3
        vol_t = F.grid_sample(vol_t, coord_grid, align_corners=True, mode='bilinear')

        xy = torch.cat([F.pad(vol.mean(dim=4), (pad, pad, pad, pad), mode='constant', value=0),
                        vol_t.mean(dim=4)], dim=1)
        xz = torch.cat([F.pad(vol.mean(dim=3), (pad, pad, pad, pad), mode='constant', value=0),
                        vol_t.mean(dim=3)], dim=1)
        yz = torch.cat([F.pad(vol.mean(dim=2), (pad, pad, pad, pad), mode='constant', value=0),
                        vol_t.mean(dim=2)], dim=1)

        downs = [8, 4, 2, 1]
        emb = torch.cat([emb, emb, emb], dim=0)
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[0](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[0](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(torch.cat([yz, xz, xy], dim=0), emb).chunk(dim=0, chunks=3)
        return self.output_xy_block(torch.cat([yz, xz, xy], dim=0))

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
        if not hasattr(self, 'grid') or self.grid.size(0) < batchsize:
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
            setattr(self, 'grid', t_grid.to(device))
        return self.grid[:batchsize]


class TriplaneGenerator12(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        self.size = 64
        self.before_ch = model_channels
        aabb_scale = 1.0
        emb_channels = 0
        self.whole_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.25 * aabb_scale, -1.25 * aabb_scale, -1.25 * aabb_scale],
                 [1.25 * aabb_scale, 1.25 * aabb_scale, 0.25 * aabb_scale]],
                dtype=torch.float32
            )
        )
        self.cur_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-1.00 * aabb_scale, -1.00 * aabb_scale, -1.00 * aabb_scale],
                 [1.00 * aabb_scale, 1.00 * aabb_scale, 0.00 * aabb_scale]],
                dtype=torch.float32
            )
        )
        # self.const = torch.nn.Parameter(torch.randn([3, hint_channels, self.size, self.size]))  # yz, xz, xy
        self.to3d = conv_nd(dims, hint_channels, self.size * 32, 1, padding=0)
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        resblock3d = partial(ResBlock, dropout=dropout, dims=dims + 1,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        self.input_3d_block = nn.ModuleList([])
        model_channels3d = 32
        ch = model_channels3d
        input_block_chans = [model_channels3d]
        for level, mult in enumerate(channel_mult):
            self.input_3d_block.append(resblock3d(channels=ch, out_channels=mult * model_channels3d,
                                                  emb_channels=emb_channels))
            ch = mult * model_channels3d
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_3d_block.append(resblock3d(channels=mult * model_channels3d,
                                                      out_channels=mult * model_channels3d, emb_channels=emb_channels,
                                                      down=True))

                input_block_chans.append(ch)
        self.mid_3d_block = conv_nd(dims + 1, ch, ch, 1)

        self.output_3d_block = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(2):
                ich = input_block_chans.pop()
                layer_xy = [
                    resblock3d(channels=ch + ich, out_channels=mult * model_channels3d, emb_channels=emb_channels)]
                ch = model_channels3d * mult
                if level > 0 and i == 1:
                    layer_xy.append(resblock3d(channels=mult * model_channels3d, emb_channels=emb_channels,
                                               out_channels=mult * model_channels3d, up=True))
                self.output_3d_block.append(TimestepEmbedSequential(*layer_xy))
        self.final_3d_block = conv_nd(dims + 1, ch, self.before_ch, 1)

        # fuse & triplane
        ch = model_channels
        self.input_xy_block = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        # input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            self.attn_blocks.append(nn.ModuleList([
                TriplaneAttention(in_channels=ch, n_heads=ch // 64),
            ]))
            self.input_xy_block.append(TimestepEmbedSequential(*[
                resblock(channels=ch, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
                resblock(channels=mult * model_channels, out_channels=mult * model_channels,
                         emb_channels=emb_channels),
            ]))
            ch = mult * model_channels
        output_channels = model_channels + 16
        self.output_xy_block = conv_nd(dims, ch, output_channels, 1)

    def forward(self, x, emb=None, x_t=None, emb_t=None, x2x_t=None, pad=0):
        def denormalize(x, aabb):
            aabb_lengths = aabb[1] - aabb[0]  # [3]
            return x * aabb_lengths + aabb[0]

        # 3d denoising
        # pad = 16
        # x = F.pad(x.permute(0, 1, 3, 2), (pad, pad, pad, pad), mode='constant', value=0)  # xy
        x = torch.cat([x.permute(0, 1, 3, 2)], dim=0)
        # emb = torch.cat([emb, emb_t], dim=0)
        b, _, lx, ly = x.shape
        x3d = self.to3d(x).reshape(b, -1, self.size, lx, ly).permute(0, 1, 3, 4, 2)
        # x3d = self.input_res3d_block(x3d, torch.cat([emb, emb_t], dim=0))

        outs = [x3d]
        emb3d = None
        # print(len(self.input_3d_block), len(self.output_3d_block))
        # exit(0)
        for i, module in enumerate(self.input_3d_block):
            x3d = module(x3d, emb3d)
            outs.append(x3d)
        x3d = self.mid_3d_block(x3d)
        for i, module in enumerate(self.output_3d_block):
            x3d = module(torch.cat([x3d, outs.pop()], dim=1), emb3d)
        x3d = self.final_3d_block(x3d)

        # 2d triplane fusion
        vol = x3d

        # coord_grid = self.get_grid(vol_t.size(0), (expand_size, expand_size, expand_size), minval=0.0, maxval=1.0,
        #                            device=x.device)
        # coord_grid[..., 1] = 1.0 - coord_grid[..., 1]
        # coord_grid = denormalize(coord_grid, self.whole_scene_box.aabb.to(x.device))
        # coord_grid = SceneBox.get_normalized_positions(torch.einsum('bij,bxyzj->bxyzi', x2x_t[:, :3, :3], coord_grid),
        #                                                self.cur_scene_box.aabb.to(x.device))
        # coord_grid = 2 * coord_grid - 1
        # coord_grid[..., 1] = -coord_grid[..., 1]  # N,es,es,es,3
        # vol_t = F.grid_sample(vol_t, coord_grid, align_corners=True, mode='bilinear')

        xy = F.pad(vol.mean(dim=4), (pad, pad, pad, pad), mode='constant', value=0)
        xz = F.pad(vol.mean(dim=3), (pad, pad, pad, pad), mode='constant', value=0)
        yz = F.pad(vol.mean(dim=2), (pad, pad, pad, pad), mode='constant', value=0)

        downs = [8, 4, 2, 1]
        emb = None
        for i, (f_xy, attn) in enumerate(
                zip(self.input_xy_block, self.attn_blocks)):
            x_feats = attn[0](xz, xy, transpose_x1=False, transpose_x2=False, down=downs[i])
            y_feats = attn[0](yz, xy, transpose_x1=False, transpose_x2=True, down=downs[i])
            z_feats = attn[0](yz, xz, transpose_x1=True, transpose_x2=True, down=downs[i])
            xy = xy + y_feats[1] + x_feats[1]
            yz = yz + y_feats[0] + z_feats[0]
            xz = xz + x_feats[0] + z_feats[1]
            yz, xz, xy = f_xy(torch.cat([yz, xz, xy], dim=0), emb).chunk(dim=0, chunks=3)
        return self.output_xy_block(torch.cat([yz, xz, xy], dim=0))

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
        if not hasattr(self, 'grid') or self.grid.size(0) < batchsize:
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
            setattr(self, 'grid', t_grid.to(device))
        return self.grid[:batchsize]


class TriplaneGenerator13(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, num_attn_layers=4,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        self.input_conv = conv_nd(dims, hint_channels, 32, 3, padding=1)
        self.feats_extractor = TimestepEmbedSequential(
            resblock(channels=32, out_channels=32, emb_channels=0),
            resblock(channels=32, out_channels=64, emb_channels=0),
            resblock(channels=64, out_channels=128, emb_channels=0),
            resblock(channels=128, out_channels=model_channels, emb_channels=0),
        )

        self.global_pool_list = nn.ModuleList([
            nn.Sequential(
                normalization(model_channels),
                nn.SiLU(),
                conv_nd(dims, model_channels, model_channels, 1, padding=0),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(model_channels, 64 * model_channels)
            ) for _ in range(16)
        ])

        self.attn_blocks = nn.ModuleList([
            TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
            TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
            TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
            TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64),
        ])

        time_embed_dim = model_channels * 4

        self.conv_blocks = nn.ModuleList([
            resblock(channels=model_channels, out_channels=model_channels, emb_channels=time_embed_dim),
            resblock(channels=model_channels, out_channels=model_channels, emb_channels=time_embed_dim),
            resblock(channels=model_channels, out_channels=model_channels, emb_channels=time_embed_dim),
            resblock(channels=model_channels, out_channels=model_channels, emb_channels=time_embed_dim),
        ])

        self.out = conv_nd(dims, model_channels, model_channels + 16, 1, padding=0)

    def forward(self, x, emb=None, x_t=None, emb_t=None, x2x_t=None, pad=0):
        # x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        bsz = x.size(0)
        x = self.input_conv(x)
        x = self.feats_extractor(x, emb)
        vol = 0.0
        for f in self.global_pool_list:
            z = f(x).reshape(bsz, -1, 64)
            z = F.interpolate(z, scale_factor=0.5, mode='linear')
            vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        vol = vol / len(self.global_pool_list)

        emb = torch.cat([emb, emb, emb], dim=0)
        h = torch.cat([vol.mean(dim=2), vol.mean(dim=3), vol.mean(dim=4)], dim=0)  # yx, zx, zy
        for i, (conv, attn) in enumerate(
                zip(self.conv_blocks, self.attn_blocks)):
            yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
            x_feats = attn(zx, yx, transpose_x1=True, transpose_x2=True)
            y_feats = attn(zy, yx, transpose_x1=True, transpose_x2=False)
            z_feats = attn(zy, zx, transpose_x1=False, transpose_x2=False)
            yx = yx + y_feats[1] + x_feats[1]
            zy = zy + y_feats[0] + z_feats[0]
            zx = zx + x_feats[0] + z_feats[1]
            h = torch.cat([yx, zx, zy], dim=0)
            h = conv(h, emb)
        return self.out(h)


class TriplaneGenerator15(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, num_tri_layers=4,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        time_embed_dim = model_channels * 4
        hidden = model_channels
        self.input_conv = conv_nd(dims, hint_channels, 32, 3, padding=1)
        self.feats_extractor = TimestepEmbedSequential(
            resblock(channels=32, out_channels=32, emb_channels=0),
            resblock(channels=32, out_channels=64, emb_channels=0),
            resblock(channels=64, out_channels=128, emb_channels=0),
            resblock(channels=128, out_channels=hidden, emb_channels=0),
        )

        self.global_pool_list = nn.ModuleList([
            nn.Sequential(
                normalization(hidden),
                nn.SiLU(),
                conv_nd(dims, hidden, hidden, 1, padding=0),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden, 32 * hidden)
            ) for _ in range(16)
        ])

        self.attn_blocks = nn.ModuleList([
            TriplaneAttention(in_channels=hidden, n_heads=hidden // 64)
            for _ in range(8)
        ])

        self.conv_blocks = nn.ModuleList([
            resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim)
            for _ in range(8)
        ])
        # self.conv_blocks.append(
        #     resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim, up=False)
        # )
        # self.conv_blocks.extend([
        #     resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim)
        #     for _ in range(2)
        # ])

        self.flow_predictor = TimestepEmbedSequential(
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            zero_module(conv_nd(dims, hidden, 2, 1, padding=0)),
            nn.Tanh(),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(12, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        self.out = zero_module(conv_nd(dims, hidden, 32 + 16, 1, padding=0))

    def forward(self, x, emb=None, x_t=None, emb_t=None, pose=None, pad=0):
        # x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        bsz = x.size(0)
        x = self.input_conv(x)
        x = self.feats_extractor(x, emb)

        if x.size(0) != pose.size(0):
            flow_x = self.flow_predictor(torch.cat([x, x], dim=0), self.pose_encoder(pose.type_as(x)))
        else:
            flow_x = self.flow_predictor(x, self.pose_encoder(pose.type_as(x)))

        vol = 0.0
        for f in self.global_pool_list:
            z = f(x).reshape(bsz, -1, 32)
            # z = F.interpolate(z, scale_factor=0.5, mode='linear')
            vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        vol = vol / len(self.global_pool_list)

        emb = torch.cat([emb, emb, emb], dim=0)
        h = torch.cat([vol.mean(dim=2), vol.mean(dim=3), vol.mean(dim=4)], dim=0)  # yx, zx, zy

        # vol = 0.0
        # for f in self.global_pool_list[:8]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # yx = vol.mean(dim=2)
        #
        # vol = 0.0
        # for f in self.global_pool_list[8:16]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # zx = vol.mean(dim=3)
        #
        # vol = 0.0
        # for f in self.global_pool_list[16:]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # zy = vol.mean(dim=4)
        # #
        # emb = torch.cat([emb, emb, emb], dim=0)
        # h = torch.cat([yx, zx, zy], dim=0)  # yx, zx, zy
        for i, (conv, attn) in enumerate(
                zip(self.conv_blocks, self.attn_blocks)):
            yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
            x_feats = attn(zx, yx, transpose_x1=True, transpose_x2=True)
            y_feats = attn(zy, yx, transpose_x1=True, transpose_x2=False)
            z_feats = attn(zy, zx, transpose_x1=False, transpose_x2=False)
            yx = yx + y_feats[1] + x_feats[1]
            zy = zy + y_feats[0] + z_feats[0]
            zx = zx + x_feats[0] + z_feats[1]
            h = torch.cat([yx, zx, zy], dim=0)
            h = conv(h, emb)
        return self.out(h), flow_x.permute(0, 2, 3, 1)


class TriplaneGenerator14(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, num_tri_layers=4,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        self.input_conv = conv_nd(dims, hint_channels, 32, 3, padding=1)
        self.feats_extractor = TimestepEmbedSequential(
            conv_nd(dims, 32, model_channels, 1, padding=0)
        )

        self.global_pool_list = nn.ModuleList([
            nn.Sequential(
                normalization(model_channels),
                nn.SiLU(),
                conv_nd(dims, model_channels, model_channels, 1, padding=0),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(model_channels, 32 * model_channels)
            ) for _ in range(16)
        ])

        self.attn_blocks = nn.ModuleList([
            TriplaneAttention(in_channels=model_channels, n_heads=model_channels // 64)
            for _ in range(num_tri_layers)
        ])

        time_embed_dim = model_channels * 4

        self.conv_blocks = nn.ModuleList([
            resblock(channels=model_channels, out_channels=model_channels, emb_channels=time_embed_dim)
            for _ in range(num_tri_layers)
        ])

        self.out = conv_nd(dims, model_channels, model_channels + 16, 1, padding=0)

    def forward(self, x, emb=None, x_t=None, emb_t=None, x2x_t=None, pad=0):
        # x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        bsz = x.size(0)
        x = self.input_conv(x)
        x = self.feats_extractor(x, emb)
        vol = 0.0
        for f in self.global_pool_list:
            z = f(x).reshape(bsz, -1, 32)
            # z = F.interpolate(z, scale_factor=0.5, mode='linear')
            vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        vol = vol / len(self.global_pool_list)

        # emb = torch.cat([emb, emb, emb], dim=0)
        yx, zx, zy = vol.mean(dim=2), vol.mean(dim=3), vol.mean(dim=4)  # yx, zx, zy
        for i, (conv, attn) in enumerate(
                zip(self.conv_blocks, self.attn_blocks)):
            # yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
            x_feats = attn(zx, yx, transpose_x1=True, transpose_x2=True)
            y_feats = attn(zy, yx, transpose_x1=True, transpose_x2=False)
            z_feats = attn(zy, zx, transpose_x1=False, transpose_x2=False)
            yx = yx + y_feats[1] + x_feats[1]
            zy = zy + y_feats[0] + z_feats[0]
            zx = zx + x_feats[0] + z_feats[1]
            # h = torch.cat([yx, zx, zy], dim=0)
            # h = conv(h, emb)
            yx, zx, zy = conv(yx, emb), conv(zx, emb), conv(zy, emb)
        yx, zx, zy = self.out(yx), self.out(zx), self.out(zy)
        # yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
        return yx, zx, zy


class TriplaneGenerator16(TimestepBlock):
    def __init__(self, hint_channels, model_channels, emb_channels, num_tri_layers=4,
                 dropout=0.0, channel_mult=(1, 2, 2, 4), use_checkpoint=False, use_scale_shift_norm=True, dims=2):
        super().__init__()
        resblock = partial(ResBlock, dropout=dropout, dims=dims,
                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

        time_embed_dim = model_channels * 4
        hidden = model_channels
        self.input_conv = conv_nd(dims, hint_channels, model_channels, 3, padding=1)

        self.global_pool_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_embed_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 32 * hidden)
            ) for _ in range(16)
        ])

        self.attn_blocks = nn.ModuleList([
            TriplaneAttention(in_channels=hidden, n_heads=hidden // 64)
            for _ in range(num_tri_layers)
        ])

        self.conv_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim),
                # resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim),
            )
            for _ in range(num_tri_layers)
        ])
        # self.conv_blocks.append(
        #         #     resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim, up=False)
        #         # )
        #         # self.conv_blocks.extend([
        #         #     resblock(channels=hidden, out_channels=hidden, emb_channels=time_embed_dim)
        #         #     for _ in range(2)
        #         # ])

        self.flow_feats = TimestepEmbedSequential(
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
            resblock(channels=hidden, out_channels=hidden, emb_channels=model_channels),
        )
        self.flow_pred = TimestepEmbedSequential(
            conv_nd(dims, hidden, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.occ_pred = TimestepEmbedSequential(
            conv_nd(dims, hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(time_embed_dim + 12, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        self.out = conv_nd(dims, hidden, 64 + 16, 1, padding=0)

    def forward(self, x, emb=None, x_t=None, emb_t=None, pose=None, pad=0):
        # x = F.interpolate(x, (32, 32), mode='bilinear')
        bsz = x.size(0)
        x = self.input_conv(x)
        # x = self.feats_extractor(x, emb)
        pose = pose.type_as(x)

        if x.size(0) != pose.size(0):
            pose = torch.cat([pose, torch.cat([emb, emb], dim=0)], dim=-1)
            flow_feats = self.flow_feats(torch.cat([x, x], dim=0), self.pose_encoder(pose))
        else:
            pose = torch.cat([pose, emb], dim=-1)
            flow_feats = self.flow_feats(x, self.pose_encoder(pose))

        occ_x, flow_x = self.occ_pred(flow_feats, None), self.flow_pred(flow_feats, None)

        vol = 0.0
        for f in self.global_pool_list:
            z = f(emb).reshape(bsz, -1, 32)
            vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        vol = vol / len(self.global_pool_list)

        emb = torch.cat([emb, emb, emb], dim=0)
        h = torch.cat([vol.mean(dim=2), vol.mean(dim=3), vol.mean(dim=4)], dim=0)  # yx, zx, zy

        # vol = 0.0
        # for f in self.global_pool_list[:8]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # yx = vol.mean(dim=2)
        #
        # vol = 0.0
        # for f in self.global_pool_list[8:16]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # zx = vol.mean(dim=3)
        #
        # vol = 0.0
        # for f in self.global_pool_list[16:]:
        #     z = f(x).reshape(bsz, -1, 32)
        #     # z = F.interpolate(z, scale_factor=0.5, mode='linear')
        #     vol += torch.einsum('bdz,bdyx->bdzyx', z, x)
        # vol = vol / 12.0
        # zy = vol.mean(dim=4)
        # #
        # emb = torch.cat([emb, emb, emb], dim=0)
        # h = torch.cat([yx, zx, zy], dim=0)  # yx, zx, zy
        for i, (conv, attn) in enumerate(
                zip(self.conv_blocks, self.attn_blocks)):
            yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
            x_feats = attn(zx, yx, transpose_x1=True, transpose_x2=True)
            y_feats = attn(zy, yx, transpose_x1=True, transpose_x2=False)
            z_feats = attn(zy, zx, transpose_x1=False, transpose_x2=False)
            yx = yx + y_feats[1] + x_feats[1]
            zy = zy + y_feats[0] + z_feats[0]
            zx = zx + x_feats[0] + z_feats[1]
            h = torch.cat([yx, zx, zy], dim=0)
            h = conv(h, emb)
        h = self.out(h)
        yx, zx, zy = torch.chunk(h, chunks=3, dim=0)
        return [yx, zx, zy], flow_x.permute(0, 2, 3, 1), occ_x
