from collections.abc import Sequence
from typing import Dict, List
import warnings

import numpy as np
import torch

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys

    def transform(self, results: Dict) -> Dict:

        for key in self.keys:
            data = self._fetch_data(results, key)
            if data is None:
                continue
            results[key] = self._to_tensor(data)

        return results

    def _to_tensor(self, data):
        """Convert objects of various python types to :obj:`torch.Tensor`.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not mmcv.is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f'Type {type(data)} cannot be converted to tensor.'
                'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
                '`Sequence`, `int` and `float`')

    def _fetch_data(self, results: Dict, key: str):
        # convert multi-level key to list
        key_list = key.split('.')

        # if the first key not in results, return None
        if key_list[0] not in results:
            warnings.warn(f'{self.__class__.__name__}: {key}'
                          f'is not in input dict.')
            return None

        current_item = results[key_list[0]]

        for single_level_key in key_list[1:]:
            # if current key not in current item, return None
            if single_level_key not in current_item:
                warnings.warn(f'{self.__class__.__name__}: {key} '
                              f'is not in input dict.')
                return None
            current_item = current_item[single_level_key]

        return current_item

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
