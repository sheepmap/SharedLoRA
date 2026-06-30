"""
MELoRA Conv2d mode with H-dimension fallback for SHFormer U-Net models.

This file is a standalone backup implementation. The training scripts still
import melora_utils.py unless their imports are changed explicitly.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MELoRAConv2d(nn.Module):
    """
    MELoRA wrapper for nn.Conv2d.

    In channel mode, inputs and outputs are split along the channel dimension.
    In height mode, inputs are split along H and every branch predicts the full
    output channels for its own H slice, then the slices are concatenated.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        r: List[int],
        lora_alpha: List[int],
        lora_dropout: float = 0.0,
        split_mode: str = "channel",
    ):
        super().__init__()

        if split_mode not in ("channel", "height"):
            raise ValueError(f"Unsupported split_mode: {split_mode}")

        l_num = len(r)
        if split_mode == "channel":
            assert conv.in_channels % l_num == 0, \
                f"in_channels ({conv.in_channels}) must be divisible by l_num ({l_num})"
            assert conv.out_channels % l_num == 0, \
                f"out_channels ({conv.out_channels}) must be divisible by l_num ({l_num})"
            in_seg = conv.in_channels // l_num
            out_seg = conv.out_channels // l_num
        else:
            in_seg = conv.in_channels
            out_seg = conv.out_channels

        self.conv = conv
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.l_num = l_num
        self.split_mode = split_mode

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        self.lora_dropout_layers = nn.ModuleList()
        self.scaling = []

        device = conv.weight.device
        dtype = conv.weight.dtype

        for i, rank in enumerate(r):
            if lora_dropout > 0.0:
                self.lora_dropout_layers.append(nn.Dropout2d(p=lora_dropout).to(device))
            else:
                self.lora_dropout_layers.append(nn.Identity())

            if rank > 0:
                self.lora_A.append(
                    nn.Conv2d(in_seg, rank, conv.kernel_size, conv.stride,
                              conv.padding, conv.dilation, groups=1, bias=False,
                              device=device, dtype=dtype)
                )
                self.lora_B.append(
                    nn.Conv2d(rank, out_seg, kernel_size=(1, 1), stride=(1, 1), bias=False,
                              device=device, dtype=dtype)
                )
                self.scaling.append(lora_alpha[i] / rank)

                nn.init.kaiming_uniform_(self.lora_A[-1].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[-1].weight)
            else:
                self.lora_A.append(None)
                self.lora_B.append(None)
                self.scaling.append(0.0)

        self.disable_adapters = False

    def forward(self, x):
        result = self.conv(x)

        if self.disable_adapters:
            return result

        if self.split_mode == "height":
            return result + self._height_lora_delta(x, result)

        return result + self._channel_lora_delta(x)

    def _channel_lora_delta(self, x):
        temp = []
        in_seg = self.conv.in_channels // self.l_num
        for i, rank in enumerate(self.r):
            if rank > 0:
                x_seg = x[:, i * in_seg:(i + 1) * in_seg, :, :]
                temp.append(
                    self.lora_B[i](
                        self.lora_A[i](
                            self.lora_dropout_layers[i](x_seg)
                        )
                    ) * self.scaling[i]
                )

        if not temp:
            return torch.zeros_like(self.conv(x))
        return torch.cat(temp, dim=1)

    def _height_lora_delta(self, x, result):
        temp = []
        h_chunks = torch.chunk(x, self.l_num, dim=2)
        for i, (rank, x_seg) in enumerate(zip(self.r, h_chunks)):
            if rank > 0:
                temp.append(
                    self.lora_B[i](
                        self.lora_A[i](
                            self.lora_dropout_layers[i](x_seg)
                        )
                    ) * self.scaling[i]
                )

        if not temp:
            return torch.zeros_like(result)

        delta = torch.cat(temp, dim=2)
        if delta.shape[-2:] != result.shape[-2:]:
            delta = F.interpolate(delta, size=result.shape[-2:], mode="bilinear", align_corners=False)
        return delta

    def merge(self):
        """Merge channel-split LoRA weights into the original conv weight."""
        if self.split_mode == "height":
            return

        if self.conv.groups > 1:
            return

        if self.conv.weight.size(2) == 1 and self.conv.weight.size(3) == 1:
            for i, rank in enumerate(self.r):
                if rank == 0:
                    continue
                delta = (
                    self.lora_B[i].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[i].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[i]
                out_seg = self.conv.out_channels // self.l_num
                in_seg = self.conv.in_channels // self.l_num
                self.conv.weight.data[
                    i * out_seg:(i + 1) * out_seg,
                    i * in_seg:(i + 1) * in_seg
                ] += delta
        else:
            full_in = self.conv.in_channels
            full_out = self.conv.out_channels
            for i, rank in enumerate(self.r):
                if rank == 0:
                    continue
                delta = torch.einsum(
                    'or,rihw->oihw',
                    self.lora_B[i].weight.squeeze(-1).squeeze(-1),
                    self.lora_A[i].weight,
                ) * self.scaling[i]
                out_seg = full_out // self.l_num
                in_seg = full_in // self.l_num
                self.conv.weight.data[
                    i * out_seg:(i + 1) * out_seg,
                    i * in_seg:(i + 1) * in_seg
                ] += delta

    def get_delta_weight(self) -> torch.Tensor:
        """Return the full delta weight tensor for channel-split mode."""
        full_in = self.conv.in_channels
        full_out = self.conv.out_channels
        kh, kw = self.conv.weight.size(2), self.conv.weight.size(3)
        full_delta = torch.zeros(
            full_out, full_in, kh, kw,
            device=self.conv.weight.device, dtype=self.conv.weight.dtype
        )

        if self.split_mode == "height":
            return full_delta

        out_seg = full_out // self.l_num
        in_seg = full_in // self.l_num
        for i, rank in enumerate(self.r):
            if rank == 0:
                continue
            delta = torch.einsum(
                'or,rihw->oihw',
                self.lora_B[i].weight.squeeze(-1).squeeze(-1),
                self.lora_A[i].weight,
            ) * self.scaling[i]
            full_delta[
                i * out_seg:(i + 1) * out_seg,
                i * in_seg:(i + 1) * in_seg
            ] = delta
        return full_delta


def apply_melora_to_model(
    model: nn.Module,
    r: List[int],
    lora_alpha: List[int],
    lora_dropout: float = 0.0,
    target_module_names: Optional[List[str]] = None,
    exclude_module_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Replace nn.Conv2d layers in the model with MELoRAConv2d wrappers.
    """
    exclude_names = exclude_module_names or []

    n_total = 0
    n_replaced = 0
    n_skipped = 0
    n_hsplit = 0

    for name, module in list(model.named_modules()):
        n_total += 1

        if not isinstance(module, nn.Conv2d):
            continue

        if any(s in name for s in exclude_names) or "lora_" in name:
            continue

        if target_module_names is not None:
            if not any(tgt in name for tgt in target_module_names):
                continue

        l_num = len(r)
        split_mode = "channel"
        if module.in_channels % l_num != 0 or module.out_channels % l_num != 0:
            can_hsplit = (
                module.in_channels % l_num != 0
                and module.out_channels % l_num == 0
                and len(r) > 1
            )
            if can_hsplit:
                split_mode = "height"
                n_hsplit += 1
                if verbose:
                    print(f"[MELoRA-H] H-SPLIT {name}: in={module.in_channels}, "
                          f"out={module.out_channels}, r={r}, alpha={lora_alpha}")
            else:
                if verbose:
                    print(f"[MELoRA-H] SKIP {name}: in={module.in_channels}, out={module.out_channels} "
                          f"not compatible with l_num={l_num}")
                n_skipped += 1
                continue

        parent_name = name.rsplit(".", 1)
        if len(parent_name) == 2:
            parent_path, attr = parent_name
            parent = dict(model.named_modules())[parent_path]
        else:
            parent = model
            attr = name

        melora_conv = MELoRAConv2d(module, r, lora_alpha, lora_dropout, split_mode=split_mode)
        setattr(parent, attr, melora_conv)
        n_replaced += 1

        if verbose:
            print(f"[MELoRA-H] REPLACE {name}: in={module.in_channels}, "
                  f"out={module.out_channels}, r={r}, split={split_mode}")

    if verbose:
        print(f"[MELoRA-H] Summary: total nn.Conv2d={n_total}, replaced={n_replaced}, "
              f"h_split={n_hsplit}, skipped={n_skipped}")

    return model


def set_melora_trainable(model: nn.Module) -> int:
    """
    Freeze all non-MELoRA parameters, make only lora_A and lora_B trainable.
    Returns count of trainable parameters.
    """
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    return trainable_count


def get_melora_state_dict(model: nn.Module) -> dict:
    """Return state dict containing only MELoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
