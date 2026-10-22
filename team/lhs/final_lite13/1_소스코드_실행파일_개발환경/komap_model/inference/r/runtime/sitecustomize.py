"""Loaded automatically by every Python process started with inference/r/runtime on PYTHONPATH.

Makes vendored trainers deterministic on a given GPU/driver/library stack without editing them:
cuDNN autotuning off, deterministic kernels preferred (warn-only so ops without a deterministic
implementation still run), and any later `torch.backends.cudnn.benchmark = True` is ignored.
Cross-GPU bit identity is not achievable and is not claimed.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if os.environ.get('KOMAP_NONDETERMINISTIC') != '1':
    try:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        _cudnn = torch.backends.cudnn

        class _CudnnModule(type(_cudnn)):
            def __setattr__(self, name, value):
                if name == 'benchmark':
                    value = False
                elif name == 'deterministic':
                    value = True
                super().__setattr__(name, value)

        _cudnn.__class__ = _CudnnModule
    except Exception:  # torch missing or very old: nothing to configure
        pass
