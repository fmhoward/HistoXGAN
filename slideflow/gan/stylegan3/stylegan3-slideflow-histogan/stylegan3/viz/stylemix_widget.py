# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imgui
from ..gui_utils import imgui_utils

#----------------------------------------------------------------------------

class StyleMixingWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.seed_def   = 1000
        self.seed       = self.seed_def
        self.mix_class  = -1
        self.animate    = False
        self.enables    = []
        self.mix_class  = -1
        self.mix_frac   = 0.5
        self.enable_mix_class   = False
        self.enable_mix_seed    = False

    @property
    def header(self):
        return "StyleGAN - style mixing" if self.visible else ""

    @property
    def visible(self):
        return (hasattr(self.viz, 'pkl') and self.viz.pkl)

    @property
    def mixing(self):
        return self.enable_mix_class or self.enable_mix_seed

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        num_enables = viz.result.get('num_ws', 18)
        self.enables += [True] * max(num_enables - len(self.enables), 0)


        if show and self.visible:
            self.content_height = imgui.get_text_line_height_with_spacing() * 3 + viz.spacing * 3

            pos2 = imgui.get_content_region_max()[0] - 1 - (viz.button_w * 2 + viz.spacing)
            pos1 = pos2 - imgui.get_text_line_height() - viz.spacing
            pos0 = viz.label_w

            # Class mixing
            imgui.text('Mixing')
            imgui.same_line(viz.label_w)

            with imgui_utils.grayed_out(num_ws == 0):
                _clicked, self.enable_mix_class = imgui.checkbox('Mix class', self.enable_mix_class)

            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.spacing)
            with imgui_utils.item_width(viz.font_size * 2), imgui_utils.grayed_out(not self.enable_mix_class):
                _something, self.mix_class = imgui.input_int('Class', self.mix_class, step=0)

            # Seed mixing
            imgui.text('')
            #imgui.same_line(viz.label_w + viz.font_size * 10 + viz.spacing)
            imgui.same_line(viz.label_w)
            with imgui_utils.grayed_out(num_ws == 0):
                _clicked, self.enable_mix_seed = imgui.checkbox('Mix seed', self.enable_mix_seed)

            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.spacing)
            with imgui_utils.item_width(viz.font_size * 3), imgui_utils.grayed_out(not self.enable_mix_seed):
                _changed, self.seed = imgui.input_int('Seed', self.seed, step=0)

            imgui.same_line(viz.label_w + viz.font_size * 11 + viz.spacing)
            with imgui_utils.grayed_out(not self.enable_mix_seed):
                _clicked, self.animate = imgui.checkbox('Anim', self.animate)

            imgui.same_line(pos2)
            with imgui_utils.item_width(viz.button_w * 2 + viz.spacing), imgui_utils.grayed_out(num_ws == 0 or not self.mixing):
                _changed, self.mix_frac = imgui.slider_float('##mix_fraction',
                                                    self.mix_frac,
                                                    min_value=0,
                                                    max_value=1,
                                                    format='Mix %.2f')

            imgui.text('Layers')
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            for idx in range(num_enables):
                imgui.same_line(round(pos0 + (pos1 - pos0) * (idx / (num_enables - 1))))
                if idx == 0:
                    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
                with imgui_utils.grayed_out(num_ws == 0):
                    _clicked, self.enables[idx] = imgui.checkbox(f'##{idx}', self.enables[idx])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f'{idx}')
            imgui.pop_style_var(1)

            imgui.same_line(pos2)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 3)
            with imgui_utils.grayed_out(num_ws == 0):
                if imgui_utils.button('All', width=viz.button_w, enabled=(num_ws != 0)):
                    self.seed = self.seed_def
                    self.animate = False
                    self.enables = [True] * num_enables
                imgui.same_line(pos2 + viz.spacing + viz.button_w)
                if imgui_utils.button('None', width=viz.button_w, enabled=(num_ws != 0)):
                    self.seed = self.seed_def
                    self.animate = False
                    self.enables = [False] * num_enables
        else:
            self.content_height = 0


        viz.args.mix_frac = self.mix_frac if self.mixing else 0
        viz.args.mix_class = self.mix_class if self.enable_mix_class else -1
        if any(self.enables[:num_ws]):
            viz.args.stylemix_idx = [idx for idx, enable in enumerate(self.enables) if enable]
            if self.enable_mix_seed:
                viz.args.stylemix_seed = self.seed & ((1 << 32) - 1)
        if self.animate:
            self.seed += 1

#----------------------------------------------------------------------------
