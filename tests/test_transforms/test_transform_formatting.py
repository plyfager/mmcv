import pytest
import torch
import numpy as np

from mmcv.transforms import ToTensor


class TestToTensor:

    def test_init(self):
        TRANSFORM = ToTensor(keys=['img_label'])
        assert TRANSFORM.keys == ['img_label']

    def test_to_tensor(self):
        TRANSFORM = ToTensor(keys=['img_label'])

        data_tensor = torch.tensor([1, 2, 3])
        tensor_from_tensor = TRANSFORM._to_tensor(data_tensor)
        assert isinstance(tensor_from_tensor, torch.Tensor)

        data_numpy = np.array([1, 2, 3])
        tensor_from_numpy = TRANSFORM._to_tensor(data_numpy)
        assert isinstance(tensor_from_numpy, torch.Tensor)

        data_list = [1, 2, 3]
        tensor_from_list = TRANSFORM._to_tensor(data_list)
        assert isinstance(tensor_from_list, torch.Tensor)

        data_int = 1
        tensor_from_int = TRANSFORM._to_tensor(data_int)
        assert isinstance(tensor_from_int, torch.Tensor)

        data_float = 1.0
        tensor_from_float = TRANSFORM._to_tensor(data_float)
        assert isinstance(tensor_from_float, torch.Tensor)

        with pytest.raises(TypeError):
            data_str = "123"
            _ = TRANSFORM._to_tensor(data_str)

    def test_fetch_data(self):
        TRANSFORM = ToTensor(["img_label"])
        results = {"img_label": [1]}
        fetched_data = TRANSFORM._fetch_data(results, "img_label")
        assert isinstance(fetched_data, list)

        TRANSFORM = ToTensor(["instances.bbox"])
        results = {"instances": {"bbox": [0, 0, 1, 1]}}
        fetched_data = TRANSFORM._fetch_data(results, "instances.bbox")
        assert isinstance(fetched_data, list)

        TRANSFORM = ToTensor(["instances.label"])
        results = {"instances": {"bbox": [0, 0, 1, 1]}}
        fetched_data = TRANSFORM._fetch_data(results, "instances.label")
        assert fetched_data is None
        fetched_data = TRANSFORM._fetch_data(results, "object.label")
        assert fetched_data is None

    def test_transform(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label'])
        results = {'instances': {'label': [1]}, 'img_label': [1]}
        results_tensor = TRANSFORMS.transform(results)
        assert isinstance(results_tensor['instances']['label'], list)
        assert isinstance(results_tensor['img_label'], torch.Tensor)

    def test_repr(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label'])
        TRANSFORMS_str = str(TRANSFORMS)
        isinstance(TRANSFORMS_str, str)
