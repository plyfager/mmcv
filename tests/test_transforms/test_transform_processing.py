import copy

import numpy as np
import pytest

from mmcv.transforms import RandomFlip, RandomResize


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
