# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import filtered_lrelu


class TestFilteredLrelu:

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3, 16, 16), requires_grad=True)
        cls.bias = torch.randn(3, requires_grad=True)
        cls.fu = torch.randn((2, 2))
        cls.fd = torch.randn((2, 2))

    def test_filtered_lrelu_cpu(self):
        out = filtered_lrelu(self.input_tensor, bias=self.bias)
        assert out.shape == (1, 3, 16, 16)

        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            fu=self.fu,
            fd=self.fd,
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different fu
        fu = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            fu=fu,
            fd=self.fd,
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different fd
        fd = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            fu=self.fu,
            fd=fd,
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different b
        input_tensor = torch.randn((1, 4, 16, 16), requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        out = filtered_lrelu(
            input_tensor,
            bias=bias,
            fu=self.fu,
            fd=self.fd,
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 4, 16, 16)

        # test with different up
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            fu=self.fu,
            fd=self.fd,
            up=4,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 32, 32)

        # test with different down
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            fu=self.fu,
            fd=self.fd,
            up=2,
            down=4,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 8, 8)

        # test with different gain
        out1 = filtered_lrelu(self.input_tensor, bias=self.bias, gain=0.2)
        out2 = filtered_lrelu(self.input_tensor, bias=self.bias, gain=0.1)
        assert torch.allclose(out1, 2 * out2)

        # test with different slope
        out = filtered_lrelu(self.input_tensor, bias=self.bias, slope=0.2)
        assert out.shape == (1, 3, 16, 16)

        # test with different clamp
        out1 = filtered_lrelu(self.input_tensor, bias=self.bias, clamp=0.2)
        out2 = filtered_lrelu(self.input_tensor, bias=self.bias, clamp=0.1)
        assert out1.max() <= 0.2
        assert out2.max() <= 0.1

        # test with different flip_filter
        out1 = filtered_lrelu(
            self.input_tensor, bias=self.bias, flip_filter=True)
        assert out.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_filtered_lrelu_cuda(self):
        out = filtered_lrelu(self.input_tensor.cuda(), bias=self.bias.cuda())
        assert out.shape == (1, 3, 16, 16)

        out = filtered_lrelu(
            self.input_tensor.cuda(),
            bias=self.bias.cuda(),
            fu=self.fu.cuda(),
            fd=self.fd.cuda(),
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different fu
        fu = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor.cuda(),
            bias=self.bias.cuda(),
            fu=fu.cuda(),
            fd=self.fd.cuda(),
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different fd
        fd = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor.cuda(),
            bias=self.bias.cuda(),
            fu=self.fu.cuda(),
            fd=fd.cuda(),
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different b
        input_tensor = torch.randn((1, 4, 16, 16), requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        out = filtered_lrelu(
            input_tensor.cuda(),
            bias=bias.cuda(),
            fu=self.fu.cuda(),
            fd=self.fd.cuda(),
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 4, 16, 16)

        # test with different up
        out = filtered_lrelu(
            self.input_tensor.cuda(),
            bias=self.bias.cuda(),
            fu=self.fu.cuda(),
            fd=self.fd.cuda(),
            up=4,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 32, 32)

        # test with different down
        out = filtered_lrelu(
            self.input_tensor.cuda(),
            bias=self.bias.cuda(),
            fu=self.fu.cuda(),
            fd=self.fd.cuda(),
            up=2,
            down=4,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 8, 8)

        # test with different gain
        out1 = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), gain=0.2)
        out2 = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), gain=0.1)
        assert torch.allclose(out1, 2 * out2)

        # test with different slope
        out = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), slope=0.2)
        assert out.shape == (1, 3, 16, 16)

        # test with different clamp
        out1 = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), clamp=0.2)
        out2 = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), clamp=0.1)
        assert out1.max() <= 0.2
        assert out2.max() <= 0.1

        # test with different flip_filter
        out1 = filtered_lrelu(
            self.input_tensor.cuda(), bias=self.bias.cuda(), flip_filter=True)
        assert out.shape == (1, 3, 16, 16)