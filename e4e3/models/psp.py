import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from e4e3.models.encoders import psp_encoders
from e4e3.models.stylegan2.model import Generator
from e4e3.models.stylegan3.model import Generator as Generator3
from e4e3.configs.paths_config import model_paths
from slideflow.gan.stylegan3.stylegan3 import dnnlib as dnnlib_sf
from slideflow.gan.stylegan3.stylegan3 import legacy as legacy_sf
import numpy as np
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt
    
class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        print(opts.stylegan_size)
        if self.opts.model_type == "stylegan2":
            self.decoder = Generator(opts.stylegan_size, 512, 2, channel_multiplier=1)
        else:
            self.decoder = Generator3(512, 0, 512, opts.stylegan_size, 3)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            #self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            with dnnlib_sf.util.open_url(self.opts.stylegan_weights) as f:
    #if 'latent_avg' in legacy_sf.load_network_pkl(f)
    #    self.la = legacy_sf.load_network_pkl(f)['latent_avg']
                self.G_ema = legacy_sf.load_network_pkl(f)['G_ema'].to(self.opts.device)
                
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')

            with dnnlib_sf.util.open_url(self.opts.stylegan_weights) as f:
                #if 'latent_avg' in legacy_sf.load_network_pkl(f)
                #    self.la = legacy_sf.load_network_pkl(f)['latent_avg']
                self.G = legacy_sf.load_network_pkl(f)
            z_samples = np.random.RandomState(123).randn(10000, self.G['G_ema'].z_dim)
            w_samples = self.G['G_ema'].to(self.opts.device).mapping(torch.from_numpy(z_samples).to(self.opts.device), None)  # [N, L, C]
            w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
            w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
            self.latent_avg = torch.from_numpy(w_avg[0]).cpu()
            if self.encoder.style_count is not None:
                self.latent_avg = self.latent_avg.repeat(self.encoder.style_count, 1)
            self.latent_avg = self.latent_avg.to(self.opts.device)
            #ckpt = torch.load(self.opts.stylegan_weights)
            self.G_ema = self.G['G_ema'].to(self.opts.device)
            #self.decoder.load_state_dict(self.G['G_ema'].to(self.opts.device).state_dict(), strict=False)
            #self.__load_latent_avg(self.G, repeat=self.encoder.style_count)

    def forward(self, x, resize=False, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        #print(x.shape)
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0
        #print(codes.shape)
        input_is_latent = not input_code
        if randomize_noise:
            images = self.G_ema.synthesis(codes, noise_mode='random')
        else:
            images = self.G_ema.synthesis(codes, noise_mode='const')

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, None
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None