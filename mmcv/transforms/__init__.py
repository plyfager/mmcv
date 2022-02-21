# Copyright (c) OpenMMLab. All rights reserved.
from .builder import TRANSFORMS

try:
    import torch  # noqa: F401
except ImportError:
    from .processing import RandomFlip, RandomResize
    from .wrappers import ApplyToMultiple, Compose, RandomChoice, Remap
    __all__ = [
        'TRANSFORMS', 'ApplyToMultiple', 'Compose', 'RandomChoice', 'Remap',
        'RandomFlip', 'RandomResize'
    ]
else:
    from .formatting import ImageToTensor, ToTensor, to_tensor
    from .processing import RandomFlip, RandomResize
    from .wrappers import ApplyToMultiple, Compose, RandomChoice, Remap

    __all__ = [
        'TRANSFORMS', 'ApplyToMultiple', 'Compose', 'RandomChoice', 'Remap',
        'ToTensor', 'to_tensor', 'ImageToTensor', 'RandomFlip', 'RandomResize'
    ]
