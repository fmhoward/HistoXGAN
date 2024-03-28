# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import os
import re
import imgui
import json

from os.path import dirname, join, exists
from tkinter.filedialog import askopenfilename
from ..gui_utils import imgui_utils
from . import renderer
from .. import dnnlib

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class PickleWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.search_dirs    = []
        self.cur_pkl        = None
        self.user_pkl       = ''
        self.recent_pkls    = []
        self.browse_cache   = dict() # {tuple(path, ...): [dnnlib.EasyDict(), ...], ...}
        self.browse_refocus = False
        self.load('', ignore_errors=True)
        self.viz.close_gan = self.close_gan

    @property
    def header(self):
        return "StyleGAN" if self.visible else ""

    @property
    def visible(self):
        return (hasattr(self.viz, 'pkl') and self.viz.pkl)

    def add_recent(self, pkl, ignore_errors=False):
        try:
            resolved = self.resolve_pkl(pkl)
            if resolved not in self.recent_pkls:
                self.recent_pkls.append(resolved)
        except:
            if not ignore_errors:
                raise

    def file_menu_options(self):
        if imgui.menu_item('Load GAN...')[1]:
            pkl = askopenfilename()
            if pkl:
                self.load(pkl, ignore_errors=True)
        if imgui.menu_item('Close GAN')[1]:
            self.close_gan()

    def close_gan(self):
        self.cur_pkl = None
        self.viz.pkl = None
        renderer = self.viz.get_renderer('stylegan')
        renderer.reset()
        self.viz.clear_result()

    def drag_and_drop_hook(self, path, ignore_errors=False) -> bool:
        if path.endswith('pkl'):
            return self.load(path, ignore_errors=ignore_errors)
        return False

    def load(self, pkl, ignore_errors=False) -> bool:
        viz = self.viz
        success = False
        viz.clear_result()
        if hasattr(viz, 'close_slide'):
            viz.close_slide(now=False)
        viz.skip_frame() # The input field will change on next frame.
        try:
            resolved = self.resolve_pkl(pkl)
            name = resolved.replace('\\', '/').split('/')[-1]
            self.cur_pkl = resolved
            self.user_pkl = resolved
            viz.result.message = f'Loading {name}...'
            viz.defer_rendering()
            if resolved in self.recent_pkls:
                self.recent_pkls.remove(resolved)
            self.recent_pkls.insert(0, resolved)
            self.viz.pkl = self.cur_pkl
            self.viz._show_tile_preview = True

            # Load the tile_px/tile_um parameters from the training options, if present
            training_options = join(dirname(self.cur_pkl), 'training_options.json')
            gan_px = 0
            gan_um = 0
            if exists(training_options):
                with open(training_options, 'r') as f:
                    opt = json.load(f)
                if 'slideflow_kwargs' in opt:
                    gan_px = opt['slideflow_kwargs']['tile_px']
                    gan_um = opt['slideflow_kwargs']['tile_um']

            if gan_px or gan_um:
                renderer = self.viz.get_renderer('stylegan')
                renderer.gan_px = gan_px
                renderer.gan_um = gan_um
                if hasattr(self.viz, 'create_toast'):
                    self.viz.create_toast(f"Loaded GAN pkl at {pkl} (tile_px={gan_px}, tile_um={gan_um})", icon="success")
            elif hasattr(self.viz, 'create_toast'):
                self.viz.create_toast(f"Loaded GAN pkl at {pkl}; unable to detect tile_px and tile_um", icon="warn")
            success = True
        except:
            success = False
            self.cur_pkl = None
            self.user_pkl = pkl
            if pkl == '':
                viz.result = dnnlib.EasyDict(message='No network pickle loaded')
            else:
                viz.result = dnnlib.EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise
        try:
            import slideflow as sf
            self.viz._gan_config = sf.util.get_gan_config(pkl)
        except Exception:
            self.viz._gan_config = None
        self.viz._tex_obj = None
        return success

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        recent_pkls = [pkl for pkl in self.recent_pkls if pkl != self.user_pkl]
        if show and self.visible:
            self.content_height = imgui.get_text_line_height_with_spacing() + viz.spacing
            imgui.text('GAN Pickle')
            imgui.same_line(viz.label_w)
            changed, self.user_pkl = imgui_utils.input_text('##pkl', self.user_pkl, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
            if changed:
                self.load(self.user_pkl, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_pkl != '':
                imgui.set_tooltip(self.user_pkl)
            imgui.same_line()
            if imgui_utils.button('Recent...', width=viz.button_w, enabled=(len(recent_pkls) != 0)):
                imgui.open_popup('recent_pkls_popup')
            imgui.same_line()
            if imgui_utils.button('Browse...', enabled=len(self.search_dirs) > 0, width=-1):
                imgui.open_popup('browse_pkls_popup')
                self.browse_cache.clear()
                self.browse_refocus = True
        else:
            self.content_height = 0

        if imgui.begin_popup('recent_pkls_popup'):
            for pkl in recent_pkls:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.load(pkl, ignore_errors=True)
            imgui.end_popup()

        if imgui.begin_popup('browse_pkls_popup'):
            def recurse(parents):
                key = tuple(parents)
                items = self.browse_cache.get(key, None)
                if items is None:
                    items = self.list_runs_and_pkls(parents)
                    self.browse_cache[key] = items
                for item in items:
                    if item.type == 'run' and imgui.begin_menu(item.name):
                        recurse([item.path])
                        imgui.end_menu()
                    if item.type == 'pkl':
                        clicked, _state = imgui.menu_item(item.name)
                        if clicked:
                            self.load(item.path, ignore_errors=True)
                if len(items) == 0:
                    with imgui_utils.grayed_out():
                        imgui.menu_item('No results found')
            recurse(self.search_dirs)
            if self.browse_refocus:
                imgui.set_scroll_here()
                viz.skip_frame() # Focus will change on next frame.
                self.browse_refocus = False
            imgui.end_popup()

        paths = viz.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load(paths[0], ignore_errors=True)

        viz.args.pkl = self.cur_pkl

    def list_runs_and_pkls(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        pkl_regex = re.compile(r'network-snapshot-\d+\.pkl')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    if entry.is_file() and pkl_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='pkl', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

    def resolve_pkl(self, pattern):
        assert isinstance(pattern, str)
        assert pattern != ''

        # URL => return as is.
        if dnnlib.util.is_url(pattern):
            return pattern

        # Short-hand pattern => locate.
        path = _locate_results(pattern)

        # Run dir => pick the last saved snapshot.
        if os.path.isdir(path):
            pkl_files = sorted(glob.glob(os.path.join(path, 'network-snapshot-*.pkl')))
            if len(pkl_files) == 0:
                raise IOError(f'No network pickle found in "{path}"')
            path = pkl_files[-1]

        # Normalize.
        path = os.path.abspath(path)
        return path

#----------------------------------------------------------------------------
