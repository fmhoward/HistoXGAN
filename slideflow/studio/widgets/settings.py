import imgui
import slideflow as sf

from ..gui import imgui_utils, theme

#----------------------------------------------------------------------------

class SettingsWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.fps_limit      = 60
        self.use_fps_limit  = False
        self.use_vsync      = True
        self.ignore_jpg     = viz._use_model_img_fmt
        self.low_memory     = viz.low_memory
        self.use_bounds     = sf.slide_backend() == 'libvips'
        self.themes         = theme.list_themes()
        self._theme_idx     = self.themes.index("Studio Dark")

        viz.set_fps_limit(self.fps_limit if self.use_fps_limit else None)
        viz.set_vsync(self.use_vsync)
        viz.low_memory = self.low_memory
        viz._use_model_img_fmt = not self.ignore_jpg


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:

            viz.header("Settings")

            _clicked, self.low_memory = imgui.checkbox('Low memory mode', self.low_memory)
            if _clicked:
                viz.low_memory = self.low_memory

            _clicked, self.use_vsync = imgui.checkbox('Vertical sync', self.use_vsync)
            if _clicked:
                viz.set_vsync(self.use_vsync)

            _clicked, self.ignore_jpg = imgui.checkbox('Ignore compression', self.ignore_jpg)
            if _clicked:
                viz._use_model_img_fmt = not self.ignore_jpg

            with imgui_utils.grayed_out(sf.slide_backend() != 'libvips'):
                _clicked, _new_val = imgui.checkbox('Use slide bounding boxes', self.use_bounds)
            if _clicked and sf.slide_backend() == 'libvips':
                self.use_bounds = _new_val
                viz._reload_wsi()
            if imgui.is_item_hovered():
                if sf.slide_backend() == 'libvips':
                    imgui.set_tooltip("Use slide bounding boxes, if present, to crop the slide images.")
                else:
                    imgui.set_tooltip("Slide bounding boxes requires libvips.")

            _clicked, self.use_fps_limit = imgui.checkbox('Limit FPS', self.use_fps_limit)
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size * 6):
                _changed, self.fps_limit = imgui.input_int('##fps_limit', self.fps_limit, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)
                if _changed or _clicked:
                    viz.set_fps_limit(self.fps_limit if self.use_fps_limit else None)

            with imgui_utils.item_width(viz.font_size*6):
                _clicked, self._theme_idx = imgui.combo(
                    "Theme",
                    self._theme_idx,
                    self.themes)
                if _clicked:
                    viz.theme = theme.get_theme(self.themes[self._theme_idx])


#----------------------------------------------------------------------------
