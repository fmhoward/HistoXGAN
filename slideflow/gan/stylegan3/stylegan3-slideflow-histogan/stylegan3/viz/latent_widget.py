# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import imgui
from .. import dnnlib
from ..gui_utils import imgui_utils

#----------------------------------------------------------------------------

class LatentWidget:

    def __init__(self, viz):
        self.viz        = viz
        self.latent     = dnnlib.EasyDict(x=0, y=0, anim=False, speed=0.25)
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.class_idx  = -1
        self.step_y     = 100
        self.viz.latent_widget = self

    @property
    def header(self):
        return "StyleGAN" if self.visible else ""

    @property
    def visible(self):
        return (hasattr(self.viz, 'pkl') and self.viz.pkl)

    def drag(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.font_size * 4e-2
        self.latent.y += dy / viz.font_size * 4e-2

    def set_seed(self, seed):
        self.latent.x = seed
        self.latent.y = 0

    def set_class(self, class_idx):
        self.class_idx = class_idx

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show and self.visible:
            self.content_height = imgui.get_text_line_height_with_spacing() + viz.spacing
            imgui.text('Latent')
            imgui.same_line(viz.label_w)
            seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            with imgui_utils.item_width(viz.font_size * 3):
                changed, seed = imgui.input_int('##seed', seed, step=0)
                if changed:
                    self.set_seed(seed)
            imgui.same_line(viz.label_w + viz.font_size * 3 + viz.spacing)
            frac_x = self.latent.x - round(self.latent.x)
            frac_y = self.latent.y - round(self.latent.y)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.latent.x += new_frac_x - frac_x
                    self.latent.y += new_frac_y - frac_y
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)


            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing * 2 + viz.button_w + viz.spacing)
            with imgui_utils.item_width(viz.font_size * 2):
                _something, self.class_idx = imgui.input_int('Class', self.class_idx, step=0)
                viz.args.class_idx = self.class_idx


            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)
            _clicked, self.latent.anim = imgui.checkbox('Anim', self.latent.anim)
            imgui.same_line()
            snapped = dnnlib.EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
            if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.latent != snapped)):
                self.latent = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.latent != self.latent_def)):
                self.latent = dnnlib.EasyDict(self.latent_def)
        else:
            self.content_height = 0

        if self.latent.anim:
            self.latent.x += viz.frame_delta * self.latent.speed
        viz.args.w0_seeds = [] # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(self.latent.x) + ofs_x
            seed_y = np.floor(self.latent.y) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            if weight > 0:
                viz.args.w0_seeds.append([seed, weight])

#----------------------------------------------------------------------------
