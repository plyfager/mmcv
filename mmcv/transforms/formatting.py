import warnings
from collections.abc import Sequence
from typing import Dict, List, Union, Tuple

import numpy as np
import torch

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
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
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys

    def transform(self, results: Dict) -> Dict:
        """Transform function to convert data to `torch.Tensor`.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: `keys` in results will be updated.
        """
        for key in self.keys:
            data = self._fetch_data(results, key)
            if data is None:
                continue
            results[key] = to_tensor(data)

        return results

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


@TRANSFORMS.register_module()
class ImageToTensor(BaseTransform):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).
    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys: Dict) -> None:
        self.keys = keys

    def transform(self, results: Dict) -> Dict:
        """Transform function to convert image in results to
        :obj:`torch.Tensor` and transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.
        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints. Whether to flip the image will follow
    the priorities below:

     - if `override` is `False`:
         - if the input dict contains the key "flip" and "flip" is set `True`,
           then flip the image.
         - flip the image according to `prob`.
     - if `override` is `True`:
         - flip the image according to `prob`.

     How to choose the "flip_direction" will follow the priorities below:

     - if `override` is `False`:
         - if the input dict contains the key "flip_direction", then flip the
           image with "flip_direction".
         - choose the "flip_direction" according to the probabilities defined
           in `prob`.
     - if `override` is `True`:
         - choose the "flip_direction" according to the probabilities defined
           in `prob`.

    If `override` is True, there are 3 flip modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.
     Args:
         prob (float | list[float], optional): The flipping probability.
             Default: None.
         direction(str | list[str], optional): The flipping direction. Options
             are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction.
         override (bool): Whether to override flip and flip_direction so as
             to call flip twice. Defaults to False.
    """

    def __init__(self,
                 prob: Union[float, List(float), None] = None,
                 direction: Union[str, List(str)] = 'horizontal',
                 override: bool = False):
        if isinstance(prob, list):
            assert mmcv.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        elif prob is None:
            pass
        else:
            raise ValueError('probs must be None, float or list of float')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(prob, list):
            assert len(self.prob) == len(self.direction)

        self.override = override

    def bbox_flip(self, bboxes: np.ndarray, img_shape: Tuple[int],
                  direction: str) -> None:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def keypoints_flip(self, keypoints: np.ndarray, img_shape: Tuple[int],
                       direction: str) -> None:
        """Flip keypoints horizontally, vertically or diagnally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        assert keypoints.shape[-1] % 2 == 0
        flipped = keypoints.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction, list):
            # None means non-flip
            direction_list = self.direction + [None]
        else:
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        else:
            non_prob = 1 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: Dict) -> None:
        """Resize images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self.bbox_flip(results['gt_bboxes'],
                                                  img_shape,
                                                  results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self.keypoints_flip(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip segs
        if results.get('gt_semantic_seg', None) is not None:
            results['gt_semantic_seg'] = mmcv.imflip(
                results['gt_semantic_seg'],
                direction=results['flip_direction'])

    def _flip_with_override(self, results: Dict) -> None:
        """Function to flip flip images, bounding boxes, semantic segmentation
        map and keypoints, when `override` is set to `True`"""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: Dict) -> Dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_semantic_seg',
                'gt_keypoints', 'flip', and 'flip_direction' keys are
                updated in result dict.
        """
        if self.override:
            self._flip_with_override(results)
        else:
            if 'flip' in results and results['flip']:
                if 'flip_direction' in results and results[
                        'flip_direction'] is not None:
                    self._flip(results)
                else:
                    self._flip_with_override(results)
            else:
                self._flip_with_override(results)

        return results

    def __repr__(self) -> None:
        return self.__class__.__name__ + f'(prob={self.prob})'


@TRANSFORMS.register_module()
class RandomResize(BaseTransform):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the first value of the target scale is
      sampled from [`scale[0][0]`, `scale[1][0]`] uniformally and the second
      value of the target scale is sampled from [`scale[0][1]`, `scale[1][1]`]
      uniformally.
    - if `scale` is a tuple, the first and second values of the target scale
      is equal to the first and second values of `scale` multiplied by a value
      sampled from [`ratio_range[0]`, `ratio_range[1]`] uniformally.

    Args:
        scale (tuple or list[tuple]): Images scales for resizing.
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): How to interpolate the original image when
            resizing. Defaults to 'bilinear'.
    """

    def __init__(self,
                 scale: Union[tuple, List[tuple]] = None,
                 ratio_range: Tuple[float] = None,
                 keep_ratio: bool = True,
                 override: bool = False,
                 bbox_clip_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:

        assert scale is not None

        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.backend = backend
        self.interpolation = interpolation

    @staticmethod
    def random_sample(scales: List[tuple]) -> tuple:
        """Randomly sample an scale is a list of tuple.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.
        Returns:
            tuple: Returns the target scale.
        """

        assert mmcv.is_list_of(scales, tuple) and len(scales) == 2
        scale_long = [max(s) for s in scales]
        scale_short = [min(s) for s in scales]
        long_edge = np.random.randint(min(scale_long), max(scale_long) + 1)
        short_edge = np.random.randint(min(scale_short), max(scale_short) + 1)
        scale = (long_edge, short_edge)
        return scale

    @staticmethod
    def random_sample_ratio(scale: tuple, ratio_range: Tuple[float]) -> tuple:
        """Randomly sample an scale is a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.
        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.
        Returns:
            tuple: Returns the target scale.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    def _random_scale(self, results: Dict) -> None:
        """Randomly sample an scale according to the type of `scale`.

        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: One new key 'scale`is added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if isinstance(self.scale, tuple):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self.random_sample_ratio(self.scale, self.ratio_range)
        elif mmcv.is_list_of(self.scale, tuple):
            scale = self.random_sample(self.scale)
        else:
            raise NotImplementedError

        results['scale'] = scale

    def _resize_img(self, results: Dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)

            results['height'], results['width'] = img.shape[:2]
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: Dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * results['scale_factor']
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, results['width'])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['height'])
            results['gt_bboxes'] = bboxes

    def _resize_keypoints(self, results: Dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints'] * results['scale_factor'][:2]
            if self.bbox_clip_border:
                keypoints[:, 0::2] = np.clip(keypoints[:, 0::2], 0,
                                             results['width'])
                keypoints[:, 1::2] = np.clip(keypoints[:, 1::2], 0,
                                             results['height'])
            results['gt_keypoints'] = keypoints

    def _resize_seg(self, results: Dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_semantic_seg', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_semantic_seg'],
                    results['scale'],
                    interpolation=self.interpolation,
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_semantic_seg'],
                    results['scale'],
                    interpolation=self.interpolation,
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_semantic_seg',
                'gt_keypoints', 'scale', 'scale_factor', 'height', 'width',
                and 'keep_ratio' keys are updated in result dict.
        """
        if self.override:
            self._random_scale(results)
        else:
            if 'scale' in results:
                pass
            elif 'scale' not in results and 'scale_factor' in results:
                h, w = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                results['scale'] = (int(scale_factor[0] * w + 0.5),
                                    int(scale_factor[1] * h + 0.5))
            else:
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'override={self.override}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
