"""EvolvedCNN: PhaseCNN warm-started from a hundred_v1 image-mIoU top-10 parent, extended with
8 togglable techniques (light/heavy) targeting boundary quality and complex-texture (eutectic Si)
regions. Every new submodule lives under `additions.*` and is zero-initialized at its output, so a
freshly-built model reproduces the parent's behaviour exactly before any training (safe warm start,
matching hundred_v1's `additions` convention in experiments/hundred_v1/hundredlab/models.py)."""
import torch
from torch import nn
from torch.nn import functional as F

from alphase.models.cnn import PhaseCNN
from alphase.models.hybrid import ResidualBlock
from alphase.data.palette import PALETTE
from alphase.data.splits import select_rows
from alphase.utils import sha256


def zero(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


LEVELS = {
    'hires_decoder': dict(light=dict(width=40, blocks=1), heavy=dict(width=64, blocks=3)),
    'hires_cnn_branch': dict(light=dict(width=24, extra_layers=2, dilated=False), heavy=dict(width=48, extra_layers=4, dilated=True)),
    'boundary_aux_plus': dict(light=dict(width=32, blocks=1), heavy=dict(width=48, blocks=2)),
    'fullres_refine': dict(light=dict(width=40, blocks=1, dilated=False), heavy=dict(width=64, blocks=2, dilated=True)),
    'composition_head': dict(light=dict(feedback=False), heavy=dict(feedback=True)),
    'texture_branch': dict(light=dict(rates=(1, 2, 4), stats=False), heavy=dict(rates=(1, 2, 4), stats=True)),
    'complex_expert': dict(light=dict(width=32, blocks=1), heavy=dict(width=48, blocks=2)),
}
# Technique 2 (crop-mix) lives entirely in sampling.mix / evoacclab/dataset.py; it has no model module.


def technique_level(techniques, key):
    item = techniques.get(key)
    if not item or not item.get('enabled'):
        return None
    level = item.get('level', 'light')
    if level not in ('light', 'heavy'):
        raise ValueError(f'Unknown technique level: {key}={level}')
    return level


class HiresDecoderRefine(nn.Module):
    """Technique 1: extra residual capacity at native (stride-1) resolution, additive residual so
    the base decoder output is preserved unless this branch learns something useful."""
    def __init__(self, channels, width, blocks):
        super().__init__()
        layers = [ResidualBlock(channels, width)] + [ResidualBlock(width, width) for _ in range(blocks - 1)]
        self.body = nn.Sequential(*layers)
        self.project = zero(nn.Conv2d(width, channels, 1))

    def forward(self, x):
        return self.project(self.body(x))


class HiresCNNBranch(nn.Module):
    """Technique 3: a dedicated full-resolution conv branch reading the raw detail input directly.
    No DINOv3 checkpoint is available on this machine (see plan); this keeps the "high-resolution
    companion" idea using only a plain CNN path."""
    def __init__(self, in_channels, out_channels, width, extra_layers, dilated):
        super().__init__()
        layers = [nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU()]
        for i in range(extra_layers):
            dilation = 2 if dilated and i == extra_layers - 1 else 1
            layers += [nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.project = zero(nn.Conv2d(width, out_channels, 1))

    def forward(self, detail):
        return self.project(self.body(detail))


class BoundaryRefine(nn.Module):
    """Technique 4: sharpens the features feeding the boundary auxiliary head."""
    def __init__(self, channels, width, blocks):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.GELU()]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.project = zero(nn.Conv2d(width, channels, 1))

    def forward(self, x):
        return self.project(self.body(x))


class FullresRefine(nn.Module):
    """Technique 5: residual logit correction computed from native-resolution decoder features
    plus the raw detail input, i.e. a lightweight learned guided-filter for the classifier logits."""
    def __init__(self, feature_channels, input_channels, out_classes, width, blocks, dilated):
        super().__init__()
        layers = [nn.Conv2d(feature_channels + input_channels, width, 3, padding=1), nn.GELU()]
        for i in range(blocks - 1):
            dilation = 2 if dilated and i == blocks - 2 else 1
            layers += [nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.project = zero(nn.Conv2d(width, out_classes, 1))

    def forward(self, features, detail):
        return self.project(self.body(torch.cat([features, detail], 1)))


class CompositionHead(nn.Module):
    """Technique 6: local phase-composition (area-fraction) head at stride 4, supervised by a
    soft-target CE against pooled ground truth (evoacclab/losses.py::_composition_term). Heavy
    level additionally feeds the predicted composition back into the stride-2 decoder feature."""
    def __init__(self, in_channels, classes, feedback_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels, classes, 1)
        self.feedback = zero(nn.Conv2d(classes, feedback_channels, 1)) if feedback_channels else None


def local_texture_stats(gray, window=7):
    """[B,1,H,W] -> [B,2,H,W]: local variance and local gradient magnitude, cheap and
    differentiable, used as explicit evidence for mottled/complex ("leopard-pattern") textures."""
    mean = F.avg_pool2d(gray, window, stride=1, padding=window // 2, count_include_pad=False)
    mean_sq = F.avg_pool2d(gray * gray, window, stride=1, padding=window // 2, count_include_pad=False)
    variance = (mean_sq - mean * mean).clamp_min(0)
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1))
    dy = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 0, 0, 1))
    gradient = (dx * dx + dy * dy).clamp_min(0).sqrt()
    return torch.cat([variance, gradient], 1)


class TextureBranch(nn.Module):
    """Technique 7: multi-dilation context at the bottleneck (ASPP-style), optionally augmented
    with explicit local-texture statistics computed from the raw image."""
    def __init__(self, channels, width, rates, use_stats):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=r, dilation=r, groups=width), nn.GELU())
            for r in rates
        ])
        self.project = zero(nn.Conv2d(len(rates) * width, channels, 1))
        self.stats_project = zero(nn.Conv2d(2, channels, 1)) if use_stats else None

    def forward(self, x, detail):
        z = self.reduce(x)
        z = torch.cat([branch(z) for branch in self.branches], 1)
        out = self.project(z)
        if self.stats_project is not None:
            stats = F.adaptive_avg_pool2d(local_texture_stats(detail[:, :1]), x.shape[-2:])
            out = out + self.stats_project(stats)
        return x + out


class ComplexExpert(nn.Module):
    """Technique 8: a small expert decoder head specialized for complex/eutectic regions, blended
    with the main classifier logits via a learned per-pixel gate (gate starts near 0, i.e. trusts
    the parent's main path at init). If technique 6 is also enabled, the gate additionally sees the
    predicted local composition."""
    def __init__(self, channels, classes, width, blocks, use_composition, composition_classes):
        super().__init__()
        layers = [ResidualBlock(channels, width)] + [ResidualBlock(width, width) for _ in range(blocks - 1)]
        self.body = nn.Sequential(*layers)
        self.expert_head = zero(nn.Conv2d(width, classes, 1))
        self.use_composition = use_composition
        gate_in = channels + (composition_classes if use_composition else 0)
        self.gate = nn.Conv2d(gate_in, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, features, gate_input):
        expert_logits = self.expert_head(self.body(features))
        gate = self.gate(gate_input).sigmoid()
        return expert_logits, gate


class EvolvedCNN(PhaseCNN):
    def __init__(self, options):
        super().__init__(3)
        techniques = options.get('techniques', {})
        for key in techniques:
            if key not in LEVELS:
                raise ValueError(f'Unknown technique: {key}')
        self.enabled = {key: technique_level(techniques, key) for key in LEVELS}
        self.additions = nn.ModuleDict()

        level = self.enabled['hires_decoder']
        if level:
            p = LEVELS['hires_decoder'][level]
            self.additions['hires_decoder'] = HiresDecoderRefine(32, p['width'], p['blocks'])

        level = self.enabled['hires_cnn_branch']
        if level:
            p = LEVELS['hires_cnn_branch'][level]
            self.additions['hires_cnn_branch'] = HiresCNNBranch(3, 32, p['width'], p['extra_layers'], p['dilated'])

        level = self.enabled['boundary_aux_plus']
        if level:
            p = LEVELS['boundary_aux_plus'][level]
            self.additions['boundary_refine'] = BoundaryRefine(32, p['width'], p['blocks'])

        level = self.enabled['fullres_refine']
        if level:
            p = LEVELS['fullres_refine'][level]
            self.additions['fullres_refine'] = FullresRefine(32, 3, 4, p['width'], p['blocks'], p['dilated'])

        level = self.enabled['composition_head']
        if level:
            p = LEVELS['composition_head'][level]
            self.additions['composition_head'] = CompositionHead(128, 4, 64 if p['feedback'] else 0)

        level = self.enabled['texture_branch']
        if level:
            p = LEVELS['texture_branch'][level]
            self.additions['texture_branch'] = TextureBranch(384, 64, p['rates'], p['stats'])

        level = self.enabled['complex_expert']
        if level:
            p = LEVELS['complex_expert'][level]
            use_composition = self.enabled['composition_head'] is not None
            self.additions['complex_expert'] = ComplexExpert(32, 4, p['width'], p['blocks'], use_composition, 4)

    def forward(self, detail, dino=None, box=None):
        if any(s % 32 for s in detail.shape[-2:]):
            raise ValueError('Pad detail inputs to multiples of 32')
        c1, c2, c4, c8, c16 = self.detail(detail)
        if 'texture_branch' in self.additions:
            c16 = self.additions['texture_branch'](c16, detail)

        def up(x, skip, block):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))

        d8 = up(c16, c8, self.up8)
        d4 = up(d8, c4, self.up4)

        composition_logits, comp_soft = None, None
        if 'composition_head' in self.additions:
            composition_logits = self.additions['composition_head'].head(d4)
            comp_soft = composition_logits.softmax(1)

        d2 = up(d4, c2, self.up2)
        if comp_soft is not None and self.additions['composition_head'].feedback is not None:
            injected = self.additions['composition_head'].feedback(
                F.interpolate(comp_soft, size=d2.shape[-2:], mode='bilinear', align_corners=False))
            d2 = d2 + injected

        d1 = up(d2, c1, self.up1)
        if 'hires_decoder' in self.additions:
            d1 = d1 + self.additions['hires_decoder'](d1)
        if 'hires_cnn_branch' in self.additions:
            d1 = d1 + self.additions['hires_cnn_branch'](detail)

        boundary_input = d1
        if 'boundary_refine' in self.additions:
            boundary_input = d1 + self.additions['boundary_refine'](d1)

        logits = self.classifier(d1)
        if 'fullres_refine' in self.additions:
            logits = logits + self.additions['fullres_refine'](d1, detail)

        if 'complex_expert' in self.additions:
            expert = self.additions['complex_expert']
            gate_input = d1
            if expert.use_composition and comp_soft is not None:
                gate_input = torch.cat([d1, F.interpolate(comp_soft, size=d1.shape[-2:], mode='bilinear', align_corners=False)], 1)
            expert_logits, gate = expert(d1, gate_input)
            logits = (1 - gate) * logits + gate * expert_logits

        out = dict(logits=logits, aux_logits=self.aux_head(d4), boundary_logits=self.boundary_head(boundary_input))
        if composition_logits is not None:
            out['composition_logits'] = composition_logits
        return out


def build_model(cfg, load_pretrained=True):
    model = EvolvedCNN(cfg['model'])
    ref = cfg['hundred']
    if load_pretrained and ref['parent_checkpoint']:
        path = ref['parent_checkpoint']
        if sha256(path) != ref['parent_sha256']:
            raise ValueError('Parent checkpoint changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        train, valid = select_rows(cfg, 'official')
        if set(state['train_ids']) != {r['image_id'] for r in train} or set(state['validation_ids']) != {r['image_id'] for r in valid}:
            raise ValueError('Parent split mismatch')
        if state['palette'] != PALETTE.tolist() or state['resolved_config']['data']['classes'] != cfg['data']['classes']:
            raise ValueError('Parent class mapping mismatch')
        selected = state['ema_model'] if state['selected_weights'] == 'ema' else state['model']
        # Parents may carry their own hundred_v1-style additions (e.g. an aspp adapter); only the
        # shared PhaseCNN backbone/decoder/heads transfer, never a parent's own addition modules.
        base_only = {k: v for k, v in selected.items() if not k.startswith('additions.')}
        missing, unexpected = model.load_state_dict(base_only, strict=False)
        if set(missing) != {n for n in model.state_dict() if n.startswith('additions.')} or unexpected:
            raise ValueError('Unexpected parent transfer')
    configure_trainable(model, cfg, cfg['train']['stages'][0])
    return model


def configure_trainable(model, cfg, stage):
    policy = stage.get('policy', 'all')
    for name, p in model.named_parameters():
        enabled = (policy == 'all' or policy == 'new' and name.startswith('additions.')
                   or policy == 'decoder' and not name.startswith(('detail.', 'encoder.'))
                   or policy == 'head' and name.startswith('classifier.'))
        if name.startswith('aux_head.') and not cfg['loss']['auxiliary']:
            enabled = False
        if name.startswith('boundary_head.') and not cfg['loss']['boundary']:
            enabled = False
        p.requires_grad_(enabled)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError('No trainable parameters')
