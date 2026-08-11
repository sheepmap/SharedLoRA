"""Minimal CPU tests for the convolutional DoRA and PiSSA adapters."""
import copy
import unittest

import torch
import torch.nn as nn

from MC_DDPM_SH.models.conv_peft import ConvDoRA, ConvPiSSA, set_conv_peft_trainable


class ConvPEFTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def _exercise_adapter(self, adapter_cls, kwargs, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        base = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        base_reference = copy.deepcopy(base).eval()
        adapter = adapter_cls(base, **kwargs).eval()
        x = torch.randn(2, in_channels, 11, 13)

        # Fresh adapters must preserve pretrained inference exactly.
        torch.testing.assert_close(adapter(x), base_reference(x), rtol=1e-5, atol=1e-6)

        set_conv_peft_trainable(adapter)
        for name, parameter in adapter.named_parameters():
            self.assertEqual(parameter.requires_grad, ('lora_' in name or 'dora_' in name), name)

        # A backward pass must reach at least one adapter parameter.
        adapter.train()
        loss = adapter(x).square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters() if parameter.requires_grad))

        # Adapter-only state restores the same adapted output on the same base.
        with torch.no_grad():
            for parameter in adapter.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01 * torch.randn_like(parameter))
        adapter.eval()
        expected = adapter(x)
        adapter_state = {
            name: parameter.detach().clone()
            for name, parameter in adapter.named_parameters() if parameter.requires_grad
        }
        restored = adapter_cls(copy.deepcopy(base_reference), **kwargs).eval()
        restored.load_state_dict(adapter_state, strict=False)
        torch.testing.assert_close(restored(x), expected, rtol=1e-5, atol=1e-6)

    def test_dora_for_3x3_and_1x1_convolutions(self):
        self._exercise_adapter(ConvDoRA, {'rank': 2, 'alpha': 4.0}, 3, 5, 3)
        self._exercise_adapter(ConvDoRA, {'rank': 1, 'alpha': 2.0}, 1, 1, 1)

    def test_pissa_for_3x3_and_1x1_convolutions(self):
        self._exercise_adapter(ConvPiSSA, {'rank': 2, 'alpha': 2.0}, 3, 5, 3)
        self._exercise_adapter(ConvPiSSA, {'rank': 4, 'alpha': 1.0}, 1, 1, 1)


if __name__ == '__main__':
    unittest.main()
