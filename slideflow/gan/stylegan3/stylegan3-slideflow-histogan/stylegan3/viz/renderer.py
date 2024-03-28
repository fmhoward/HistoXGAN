# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import copy
import traceback
import numpy as np
import torch
import torch.fft
import torch.nn
import matplotlib.cm
import threading

from ..torch_utils.ops import upfirdn2d
from .. import legacy # pylint: disable=import-error
from .. import dnnlib

from rich import print
import slideflow as sf
from slideflow.gan.utils import crop

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def get_device(device=None):
    if device is None and torch.cuda.is_available():
        return torch.device('cuda')
    elif (device is None
          and hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif device is None:
        return torch.device('cpu')
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device

#----------------------------------------------------------------------------

def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

def _reduce_dropout_preds(yp_drop, num_outcomes):
    if sf.backend() == 'tensorflow':
        import tensorflow as tf
        if num_outcomes > 1:
            yp_drop = [tf.stack(yp_drop[n], axis=0) for n in range(num_outcomes)]
            yp_mean = [tf.math.reduce_mean(yp_drop[n], axis=0)[0].numpy() for n in range(num_outcomes)]
            yp_std = [tf.math.reduce_std(yp_drop[n], axis=0)[0].numpy() for n in range(num_outcomes)]
        else:
            yp_drop = tf.stack(yp_drop[0], axis=0)
            yp_mean = tf.math.reduce_mean(yp_drop, axis=0)[0].numpy()
            yp_std = tf.math.reduce_std(yp_drop, axis=0)[0].numpy()
    else:
        if num_outcomes > 1:
            stacked = [torch.stack(yp_drop[n], dim=0) for n in range(num_outcomes)]
            yp_mean = [torch.mean(stacked[n], dim=0) for n in range(num_outcomes)]
            yp_std = [torch.std(stacked[n], dim=0) for n in range(num_outcomes)]
        else:
            stacked = torch.stack(yp_drop[0], dim=0)  # type: ignore
            yp_mean = torch.mean(stacked, dim=0)  # type: ignore
            yp_std = torch.std(stacked, dim=0)  # type: ignore
    return yp_mean, yp_std

#----------------------------------------------------------------------------

def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------

def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, visualizer=None, gan_px=0, gan_um=0):
        self._visualizer        = visualizer
        self._device            = get_device()
        self._pkl_data          = dict()    # {pkl: dict | CapturedException, ...}
        self._networks          = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs       = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps             = dict()    # {name: torch.Tensor, ...}
        self._is_timing         = False
        self._net_layers        = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}
        self._uq_thread         = None
        self._stop_uq_thread    = False
        self._stop_pred_thread  = False
        self.gan_px             = gan_px
        self.gan_um             = gan_um

        # Only record stream if the device is CUDA,
        # as the MPS device from MacOS does not yet support htis.
        if self._device.type == 'cuda':
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        else:
            self._start_event   = None
            self._end_event     = None

    def render(self, **args):
        if self._start_event is not None:
            self._is_timing = True
            self._start_event.record(torch.cuda.current_stream(self._device))
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        if self._end_event is not None:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            if not isinstance(res.image, np.ndarray):
                res.image = self.to_cpu(res.image).numpy()
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).numpy()
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def preprocess(self, img, tile_px, tile_um):
        """Preprocess a generated image (uint8) for use with a model."""
        img = sf.io.torch.whc_to_cwh(img)
        img = crop(
            img,
            gan_um=self.gan_um,
            gan_px=self.gan_px,
            target_um=tile_um,
        )
        img = sf.io.torch.preprocess_uint8(img, standardize=False, resize_px=tile_px)
        img = sf.io.torch.cwh_to_whc(img)
        return sf.io.convert_dtype(img, np.uint8)

    def reset(self):
        del self._pkl_data
        self._pkl_data = dict()

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = self._tweak_network(net, **tweak_kwargs)
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _tweak_network(self, net):
        # Print diagnostics.
        #for name, value in misc.named_params_and_buffers(net):
        #    if name.endswith('.magnitude_ema'):
        #        value = value.rsqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        #    if name.endswith('.weight') and value.ndim == 4:
        #        value = value.square().mean([1,2,3]).sqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype)
            if self._device.type == 'cuda':
                buf = buf.pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def _calc_uncertainty(self, img):
        if self._visualizer is None:
            raise ValueError("Visualizer not loaded.")
        self._visualizer._uncertainty = None
        if self._stop_uq_thread and self._uq_thread is not None:
            self._uq_thread.join()

        def calc_uq(uq_n=30):
            pred_fn = self._visualizer._model
            for i in range(uq_n):
                if self._stop_uq_thread:
                    return
                yp = pred_fn(img, training=False)
                if i == 0:
                    num_outcomes = 1 if not isinstance(yp, list) else len(yp)
                    yp_drop = {n: [] for n in range(num_outcomes)}
                if num_outcomes > 1:
                    for o in range(num_outcomes):
                        yp_drop[o] += [yp[o]]
                else:
                    yp_drop[0] += [yp]
            yp_mean, yp_std = _reduce_dropout_preds(yp_drop, num_outcomes)
            if num_outcomes > 1:
                self._visualizer._uncertainty = [np.mean(s) for s in yp_std]
            else:
                self._visualizer._uncertainty = np.mean(yp_std)
            print("UQ Predictions:", yp_mean)
            print("UQ Uncertainty:", yp_std)
            self._stop_uq_thread = True

        self._stop_uq_thread = False
        self._uq_thread = threading.Thread(target=calc_uq)
        self._uq_thread.start()

    def _render_impl(self, res,
        pkl             = None,
        w0_seeds        = [[0, 1]],
        stylemix_idx    = [],       # Specify layers of target seed to mix
        stylemix_seed   = None,        # Specify target seed from which to mix layers
        class_idx       = None,
        mix_class       = None,
        mix_frac        = 1,
        trunc_psi       = 1,
        trunc_cutoff    = 0,
        random_seed     = 0,
        noise_mode      = 'const',
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        fft_show        = False,
        fft_all         = True,
        fft_range_db    = 50,
        fft_beta        = 8,
        input_transform = None,
        untransform     = False,
        show_saliency   = False,
        saliency_overlay= False,
        saliency_method = 0,
        use_model       = True,
        **kwargs
    ):
        if pkl is None:
            return

        # Stop UQ thread if running.
        self._stop_uq_thread = True

        # Dig up network details.
        G = self.get_network(pkl, 'G_ema')
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Handle class labels.
        if mix_class is not None and mix_class < 0:
            mix_class = None
        if class_idx is not None and class_idx < 0:
            class_idx = None

        # Generate random latents.
        all_seeds = [seed for seed, _weight in w0_seeds]
        if stylemix_seed is not None:
            all_seeds += [stylemix_seed]
        all_seeds = list(set(all_seeds))
        all_zs = np.zeros([len(all_seeds), G.z_dim], dtype=np.float32)
        all_cs = np.zeros([len(all_seeds), G.c_dim], dtype=np.float32)
        all_cs_mix = np.zeros([len(all_seeds), G.c_dim], dtype=np.float32)
        for idx, seed in enumerate(all_seeds):
            rnd = np.random.RandomState(seed)
            all_zs[idx] = rnd.randn(G.z_dim)
            # Class index for target class.
            if G.c_dim > 0 and class_idx is not None:
                all_cs[idx, class_idx] = 1
                _class_idx = class_idx
            elif G.c_dim > 0:
                all_cs[idx, rnd.randint(G.c_dim)] = 1
                _class_idx = "-"
            else:
                _class_idx = None

            # Class index for style mixing.
            if G.c_dim > 0 and mix_class is not None:
                all_cs_mix[idx, mix_class] = 1
                _mix_class_idx = mix_class
            elif G.c_dim > 0:
                all_cs_mix[idx, rnd.randint(G.c_dim)] = 1
                _mix_class_idx = "-"
            else:
                _mix_class_idx = None

        # Run mapping network.
        w_avg = G.mapping.w_avg
        all_zs = self.to_device(torch.from_numpy(all_zs))
        all_cs = self.to_device(torch.from_numpy(all_cs))
        all_cs_mix = self.to_device(torch.from_numpy(all_cs_mix))
        all_ws = G.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
        all_ws = dict(zip(all_seeds, all_ws))
        all_ws_mix = G.mapping(z=all_zs, c=all_cs_mix, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
        all_ws_mix = dict(zip(all_seeds, all_ws_mix))

        # Calculate final W.
        # Size = (1, 16, px),
        #     where first dimension is the batch size,
        #     second dimension is the number of layers (selected by stylemix_idx),
        #     and final dimension is Z size.
        w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
        stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < G.num_ws]
        if len(stylemix_idx) > 0:
            w_mixed_dest_class = torch.stack([all_ws_mix[seed][stylemix_idx] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
            if mix_frac == 1:
                if stylemix_seed is not None:
                    w[:, stylemix_idx] = all_ws_mix[stylemix_seed][np.newaxis, stylemix_idx]
                else:
                    w[:, stylemix_idx] = w_mixed_dest_class
            else:
                if stylemix_seed is not None:
                    w[:, stylemix_idx] = (((all_ws_mix[stylemix_seed][np.newaxis, stylemix_idx]) * mix_frac)
                                           + (w[:, stylemix_idx] * (1-mix_frac)))
                else:
                    w[:, stylemix_idx] = (w_mixed_dest_class * mix_frac) + (w[:, stylemix_idx] * (1-mix_frac))
        w += w_avg

        # Run synthesis network.
        synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32)
        torch.manual_seed(random_seed)
        out, layers = self.run_synthesis_net(G.synthesis, w, capture_layer=layer_name, **synthesis_kwargs)

        # Update layer list.
        cache_key = (G.synthesis, tuple(sorted(synthesis_kwargs.items())))
        if cache_key not in self._net_layers:
            if layer_name is not None:
                torch.manual_seed(random_seed)
                _out, layers = self.run_synthesis_net(G.synthesis, w, **synthesis_kwargs)
            self._net_layers[cache_key] = layers
        res.layers = self._net_layers[cache_key]

        # Untransform.
        if untransform and res.has_input_transform:
            out, _mask = _apply_affine_transformation(out.to(torch.float32), G.synthesis.input.transform, amax=6) # Override amax to hit the fast path in upfirdn2d.

        # Select channels and compute statistics.
        out = out[0].to(torch.float32)
        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel : base_channel + sel_channels]
        res.stats = torch.stack([
            out.mean(), sel.mean(),
            out.std(), sel.std(),
            out.norm(float('inf')), sel.norm(float('inf')),
        ])

        # Scale and convert to uint8.
        img = sel
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        gan_out_img = img
        img = img.permute(1, 2, 0)
        res.image = img

        # FFT.
        if fft_show:
            sig = out if fft_all else sel
            sig = sig.to(torch.float32)
            sig = sig - sig.mean(dim=[1,2], keepdim=True)
            sig = sig * torch.kaiser_window(sig.shape[1], periodic=False, beta=fft_beta, device=self._device)[None, :, None]
            sig = sig * torch.kaiser_window(sig.shape[2], periodic=False, beta=fft_beta, device=self._device)[None, None, :]
            fft = torch.fft.fftn(sig, dim=[1,2]).abs().square().sum(dim=0)
            fft = fft.roll(shifts=[fft.shape[0] // 2, fft.shape[1] // 2], dims=[0,1])
            fft = (fft / fft.mean()).log10() * 10 # dB
            fft = self._apply_cmap((fft / fft_range_db + 1) / 2)
            res.image = torch.cat([img.expand_as(fft), fft], dim=1)

        # Copy Tensor to CPU.
        res.image = res.image.cpu()

        # Show predictions.
        if use_model and self._visualizer is not None and self._visualizer._use_model:
            img = gan_out_img
            if sf.backend() == 'tensorflow':
                import tensorflow as tf
                dtype = tf.uint8
                to_numpy = lambda x: x.numpy()
            elif sf.backend() == 'torch':
                dtype = torch.uint8
                to_numpy = lambda x: x.cpu().detach().numpy()

            target_px = self._visualizer.tile_px
            crop_kw = dict(
                gan_um=self.gan_um,
                gan_px=self.gan_px,
                target_um=self._visualizer.tile_um,
            )
            img = crop(img, **crop_kw)  # type: ignore
            img = sf.io.convert_dtype(img, dtype)

            # Resize.
            if sf.backend() == 'tensorflow':
                img = sf.io.tensorflow.preprocess_uint8(img, standardize=False, resize_px=target_px)['tile_image']
            else:
                img = sf.io.torch.preprocess_uint8(img, standardize=False, resize_px=target_px)

            # Normalize.
            normalizer = self._visualizer._normalizer
            if normalizer:
                img = normalizer.transform(img)
                if not isinstance(img, np.ndarray):
                    res.normalized = img.numpy().astype(np.uint8)
                else:
                    res.normalized = img.astype(np.uint8)

            # Finish pre-processing.
            if sf.backend() == 'tensorflow':
                img = sf.io.tensorflow.preprocess_uint8(img, standardize=True)['tile_image']
                img = tf.expand_dims(img, axis=0)
            elif sf.backend() == 'torch':
                img = sf.io.torch.preprocess_uint8(img, standardize=True)
                img = torch.unsqueeze(img, dim=0)
            preds = self._visualizer._model(img)
            if isinstance(preds, list):
                preds = [to_numpy(p[0]) for p in preds]
            else:
                preds = to_numpy(preds[0])
            self._visualizer._predictions = preds

            # UQ --------------------------------------------------------------
            if self._visualizer.has_uq() and self._visualizer._use_uncertainty:
                self._calc_uncertainty(img)
            # -----------------------------------------------------------------

            # Saliency.
            if show_saliency:
                mask = self._visualizer.saliency.get(img[0], method=saliency_method)
                if saliency_overlay:
                    res.image = sf.grad.plot_utils.overlay(res.image, mask)
                else:
                    res.image = sf.grad.plot_utils.inferno(mask)
                if res.image.shape[-1] == 4:
                    res.image = res.image[:, :, 0:3]

    @staticmethod
    def run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

#----------------------------------------------------------------------------
