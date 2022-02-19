import copy

import numpy as np
import pytest
import torch

from mmcv.transforms import (ImageToTensor, RandomFlip, RandomResize, ToTensor,
                             to_tensor)


def test_to_tensor():

    # The type of the input object is torch.Tensor
    data_tensor = torch.tensor([1, 2, 3])
    tensor_from_tensor = to_tensor(data_tensor)
    assert isinstance(tensor_from_tensor, torch.Tensor)

    # The type of the input object is numpy.ndarray
    data_numpy = np.array([1, 2, 3])
    tensor_from_numpy = to_tensor(data_numpy)
    assert isinstance(tensor_from_numpy, torch.Tensor)

    # The type of the input object is list
    data_list = [1, 2, 3]
    tensor_from_list = to_tensor(data_list)
    assert isinstance(tensor_from_list, torch.Tensor)

    # The type of the input object is int
    data_int = 1
    tensor_from_int = to_tensor(data_int)
    assert isinstance(tensor_from_int, torch.Tensor)

    # The type of the input object is float
    data_float = 1.0
    tensor_from_float = to_tensor(data_float)
    assert isinstance(tensor_from_float, torch.Tensor)

    # The type of the input object is invalid
    with pytest.raises(TypeError):
        data_str = '123'
        _ = to_tensor(data_str)


class TestToTensor:

    def test_init(self):
        TRANSFORM = ToTensor(keys=['img_label'])
        assert TRANSFORM.keys == ['img_label']

    def test_transform(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label', 'bbox'])

        # Test multi-level key and single-level key (multi-level key is
        # not in results)
        results = {'instances': {'label': [1]}, 'img_label': [1]}
        results_tensor = TRANSFORMS.transform(copy.deepcopy(results))
        assert isinstance(results_tensor['instances']['label'], list)
        assert isinstance(results_tensor['img_label'], torch.Tensor)

        # Test multi-level key (multi-level key is in results)
        results = {'instances': {'bbox': [1]}, 'img_label': [1]}
        results_tensor = TRANSFORMS.transform(copy.deepcopy(results))
        assert isinstance(results_tensor['instances']['bbox'], torch.Tensor)

    def test_repr(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label'])
        TRANSFORMS_str = str(TRANSFORMS)
        isinstance(TRANSFORMS_str, str)


class TestImageToTensor:

    def test_init(self):
        TRANSFORMS = ImageToTensor(['img'])
        assert TRANSFORMS.keys == ['img']

    def test_transform(self):
        TRANSFORMS = ImageToTensor(['img'])

        # image only has one channel
        results = {'img': np.zeros((224, 224))}
        results = TRANSFORMS.transform(results)
        assert results['img'].shape == (1, 224, 224)

        # image has three channels
        results = {'img': np.zeros((224, 224, 3))}
        results = TRANSFORMS.transform(results)
        assert results['img'].shape == (3, 224, 224)

    def test_repr(self):
        TRANSFORMS = ImageToTensor(['img'])
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)


class TestRandomFlip:

    def test_init(self):

        # prob is float
        TRANSFORMS = RandomFlip(0.1)
        assert TRANSFORMS.prob == 0.1

        # prob is None
        TRANSFORMS = RandomFlip(None)
        assert TRANSFORMS.prob is None

        # prob is a list
        TRANSFORMS = RandomFlip([0.1, 0.2], ['horizontal', 'vertical'])
        assert len(TRANSFORMS.prob) == 2
        assert len(TRANSFORMS.direction) == 2

        # direction is an invalid type
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(0.1, 1)

        # prob is an invalid type
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip('0.1')

    def test_transform(self):

        results = {
            'img': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_keypoints': np.array([[100, 100]]),
            'gt_semantic_seg': np.random.random((224, 224, 3))
        }

        # horizontal flip
        TRANSFORMS = RandomFlip([1.0], ['horizontal'], override=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        # diagnal flip
        TRANSFORMS = RandomFlip([1.0], ['diagonal'], override=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 123, 224,
                                                          223]])).all()

        # horizontal flip when direction is str
        TRANSFORMS = RandomFlip(1.0, override=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        # vertical flip when override is False
        results.update({'flip': True, 'flip_direction': 'vertical'})
        TRANSFORMS = RandomFlip(1.0, override=False)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[0, 123, 100,
                                                          223]])).all()

        # flip with setting from init when flip_direction is None in results
        results.update({'flip_direction': None})
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        # flip with setting from init when flip is False in results
        results.update({'flip': False})
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        # do not flip
        TRANSFORMS = RandomFlip(0.0)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[0, 1, 100,
                                                          101]])).all()

        # flip direction is invalid in bbox flip
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(1.0, override=False)
            results_update = TRANSFORMS.bbox_flip(results['gt_bboxes'],
                                                  (224, 224), 'invalid')

        # flip direction is invalid in keypoints flip
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(1.0, override=False)
            results_update = TRANSFORMS.keypoints_flip(results['gt_keypoints'],
                                                       (224, 224), 'invalid')

    def test_repr(self):
        TRANSFORMS = RandomFlip(0.1)
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)


class TestRandomResize:

    def test_init(self):
        TRANSFORMS = RandomResize(
            (224, 224),
            (1.0, 2.0),
        )
        assert TRANSFORMS.scale == (224, 224)

    def test_repr(self):
        TRANSFORMS = RandomResize(
            (224, 224),
            (1.0, 2.0),
        )
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)

    def test_transform(self):

        # choose target scale from init when override is True
        results = {}
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0), override=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'][0] >= 224 and results_update['scale'][
            0] <= 448
        assert results_update['scale'][1] >= 224 and results_update['scale'][
            1] <= 448

        # choose target scale from results when override is False
        results = {'scale': (224, 224)}
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0), override=False)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'] == (224, 224)

        # Use scale factor in results when override is False
        results = {
            'scale_factor': (1.0, 0.5, 1.0, 0.5),
            'img': np.random.random((224, 224, 3)),
            'gt_semantic_seg': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 0, 112, 112]]),
            'gt_keypoints': np.array([[112, 112]])
        }
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0),
                                  override=False,
                                  keep_ratio=False)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'] == (224, 112)
        assert results_update['height'] == 112
        assert results_update['width'] == 224
        assert results_update['gt_semantic_seg'].shape == (112, 224, 3)
        assert results_update['gt_bboxes'][0][2] == 112
        assert results_update['gt_bboxes'][0][3] == 56
        assert results_update['gt_keypoints'][0][0] == 112
        assert results_update['gt_keypoints'][0][1] == 56

        # keep ratio is True
        results = {
            'scale_factor': (1.0, 0.5, 1.0, 0.5),
            'img': np.random.random((224, 224, 3)),
            'gt_semantic_seg': np.random.random((224, 224, 3))
        }
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0),
                                  override=False,
                                  keep_ratio=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'] == (224, 112)
        assert results_update['height'] == 112
        assert results_update['width'] == 112
        assert results_update['gt_semantic_seg'].shape == (112, 112, 3)
        assert results_update['keep_ratio']

        # choose target scale from init when override is False and scale is a 
        # list of tuples
        results = {}
        TRANSFORMS = RandomResize([(224, 448), (112, 224)],
                                  override=False,
                                  keep_ratio=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'][0] >= 224 and results_update['scale'][
            0] <= 448
        assert results_update['scale'][1] >= 112 and results_update['scale'][
            1] <= 224

        # the type of scale is invalid in init
        with pytest.raises(NotImplementedError):
            results = {}
            TRANSFORMS = RandomResize([(224, 448), [112, 224]],
                                      override=False,
                                      keep_ratio=True)
            results_update = TRANSFORMS.transform(copy.deepcopy(results))
