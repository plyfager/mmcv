from typing import List, Tuple, Union

import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map.

    Added or Updated keys: flip, flip_direction, img, gt_bboxes,
    gt_semantic_seg, and gt_keypoints.

    There are 3 flip modes:

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
             Defaults to None.
         direction(str | list[str], optional): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to horizontal.
    """

    def __init__(self,
                 prob: Union[float, List[float], None] = None,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert mmcv.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        elif prob is None:
            pass
        else:
            raise ValueError(
                f"probs must be None, float or list of float, but \
                              got '{type(prob)}'.")
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f"direction must be either str or list of str, \
                               but got '{type(direction)}'.")
        self.direction = direction

        if isinstance(prob, list):
            assert len(self.prob) == len(self.direction)

    def bbox_flip(self, bboxes: np.ndarray, img_shape: Tuple[int],
                  direction: str) -> np.ndarray:
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
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagnal', but got '{direction}'")
        return flipped

    def keypoints_flip(self, keypoints: np.ndarray, img_shape: Tuple[int],
                       direction: str) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagnally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
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
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagnal', but got '{direction}'")
        flipped = np.concatenate([keypoints, meta_info], axis=-1)
        return flipped

    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction, list):
            # None means non-flip
            direction_list = self.direction + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]
        else:
            raise ValueError(f"Only support list and str, but \
                               got '{type(self.direction)}'")

        if isinstance(self.prob, list):
            non_prob = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]
        else:
            raise ValueError(f"Only support list and float, but \
                               got '{type(self.prob)}'")

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
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

    def _flip_with_flip_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_semantic_seg',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_with_flip_direction(results)

        return results

    def __repr__(self) -> None:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.prob}, '
        repr_str += f'interpolation={self.direction})'

        return repr_str


@TRANSFORMS.register_module()
class RandomResize(BaseTransform):
    """Random resize images & bbox & keypoints.

    Added or updated keys: scale, scale_factor, keep_ratio, img, height, width,
    gt_bboxes, gt_semantic_seg, and gt_keypoints.


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
            Defaults to None.
        ratio_range (tuple[float]): (min_ratio, max_ratio). Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to True.
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
                 bbox_clip_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:

        assert scale is not None

        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border
        self.backend = backend
        self.interpolation = interpolation

    @staticmethod
    def _random_sample(scales: List[tuple]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

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
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

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

    def _random_scale(self, results: dict) -> None:
        """Private function to randomly sample an scale according to the type
        of `scale`.

        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: One new key 'scale`is added into ``results``,
            which would be used by subsequent pipelines.
        """

        if isinstance(self.scale, tuple):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(self.scale, self.ratio_range)
        elif mmcv.is_list_of(self.scale, tuple):
            scale = self._random_sample(self.scale)
        else:
            raise NotImplementedError(f"Do not support sampling function \
                                        for '{self.scale}'")

        results['scale'] = scale

    def _resize_img(self, results: dict) -> None:
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

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * results['scale_factor']
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, results['width'])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['height'])
            results['gt_bboxes'] = bboxes

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints'] * results['scale_factor'][:2]
            if self.bbox_clip_border:
                keypoints[:, 0::2] = np.clip(keypoints[:, 0::2], 0,
                                             results['width'])
                keypoints[:, 1::2] = np.clip(keypoints[:, 1::2], 0,
                                             results['height'])
            results['gt_keypoints'] = keypoints

    def _resize_seg(self, results: dict) -> None:
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

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_semantic_seg',
            'gt_keypoints', 'scale', 'scale_factor', 'height', 'width',
            and 'keep_ratio' keys are updated in result dict.
        """
        self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        self._resize_seg(results)
        return results

    def __repr__(self) -> None:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
