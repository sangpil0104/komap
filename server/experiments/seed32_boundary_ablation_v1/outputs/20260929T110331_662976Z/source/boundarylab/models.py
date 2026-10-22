import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(4, cout), nn.SiLU())


class DetailBranch(nn.Module):
    """Input-resolution features; no downsampling. Zero-initialized logit residual."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv(1, 16), conv(16, 16))
        self.fuse = conv(32, 16)
        self.head = nn.Conv2d(16, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, gray, decoder_features):
        return self.head(self.fuse(torch.cat([self.stem(gray), decoder_features], 1)))


class NativeRefiner(nn.Module):
    def __init__(self, use_gray=True):
        super().__init__()
        self.use_gray = use_gray
        self.features = nn.Sequential(conv(5, 24), conv(24, 24), conv(24, 16))
        self.head = nn.Conv2d(16, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, gray, probability):
        # Gray-off arm keeps the exact architecture/parameter count.
        gray = gray if self.use_gray else torch.zeros_like(gray)
        correction = self.head(self.features(torch.cat([gray, probability], 1)))
        return (probability.clamp_min(1e-7).log() + correction.float()).softmax(1)


def restore_probs(probability, shape, mode='area'):
    if tuple(probability.shape[-2:]) == tuple(shape):
        return probability
    if mode == 'area':
        return F.interpolate(probability, size=shape, mode='area')
    if mode == 'bilinear':
        return F.interpolate(probability, size=shape, mode='bilinear', align_corners=False)
    raise ValueError(mode)


class BoundaryModel(nn.Module):
    def __init__(self, cfg, encoder_weights=None, reference_checkpoint=None):
        super().__init__()
        if encoder_weights is not None and reference_checkpoint is not None:
            raise ValueError('Choose ImageNet initialization OR read-only reference evaluation')
        self.cfg = cfg
        self.base = smp.Unet(encoder_name='efficientnet-b4', encoder_weights=None, in_channels=3, classes=4)
        if encoder_weights is not None:
            self.base.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu', weights_only=True))
        if reference_checkpoint is not None:
            self.base.load_state_dict(torch.load(reference_checkpoint, map_location='cpu', weights_only=True), strict=True)
        for block in self.base.decoder.blocks:
            block.interpolation_mode = cfg['decoder_interpolation']
        # SMP retains the classifier's unused 1x1 head. It never enters encoder.forward.
        for name in ('_conv_head', '_bn1'):
            for parameter in getattr(self.base.encoder, name).parameters():
                parameter.requires_grad_(False)
        self.detail_branch = DetailBranch() if cfg['detail_branch'] else None
        self.native_refiner = NativeRefiner(cfg['refiner_use_gray']) if cfg['native_refiner'] else None

    def detail_logits(self, detail):
        features = self.base.decoder(self.base.encoder(detail))
        logits = self.base.segmentation_head(features)
        if self.detail_branch is not None:
            logits = logits + self.detail_branch(detail[:, :1], features)
        return logits

    def refine(self, native_gray, probability):
        return self.native_refiner(native_gray, probability) if self.native_refiner is not None else probability

    def forward(self, detail, native_gray=None):
        logits = self.detail_logits(detail)
        native = None
        if native_gray is not None:
            native = restore_probs(logits.float().softmax(1), native_gray.shape[-2:], self.cfg['restore'])
            native = self.refine(native_gray, native)
        return dict(logits=logits, native=native)
