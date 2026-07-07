"""
MELoRA Conv2d mode for SHFormer U-Net models.
Based on the peft MELoRA implementation (melora.py:1014).
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_A_GATE_TANH_RESIDUAL = 0.1


class MELoRAConv2d(nn.Module):
    """
    MELoRA wrapper for nn.Conv2d.
    Splits in_channels and out_channels into l_num segments,
    each segment has its own rank for hierarchical low-rank decomposition.

    output = W * x + sum_i( B_i(A_i(x_seg_i)) * scaling_i )
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        r: List[int],
        lora_alpha: List[int],
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        l_num = len(r)
        assert conv.in_channels % l_num == 0, \
            f"in_channels ({conv.in_channels}) must be divisible by l_num ({l_num})"
        assert conv.out_channels % l_num == 0, \
            f"out_channels ({conv.out_channels}) must be divisible by l_num ({l_num})"

        # Keep original conv frozen
        self.conv = conv
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.l_num = l_num

        in_seg = conv.in_channels // l_num
        out_seg = conv.out_channels // l_num

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
        self.a_gate_index = None
        self.b_gate_index = None
        self.current_a_gate = None
        self.current_b_gate = None

    def forward(self, x):
        result = self.conv(x)

        if self.disable_adapters:
            return result

        temp = []
        in_seg = self.conv.in_channels // self.l_num
        for i, rank in enumerate(self.r):
            if rank > 0:
                x_seg = x[:, i * in_seg:(i + 1) * in_seg, :, :]
                a_out = self.lora_A[i](self.lora_dropout_layers[i](x_seg))
                if self.current_a_gate is not None:
                    a_gate = self.current_a_gate.to(device=a_out.device, dtype=a_out.dtype)
                    z = a_gate * a_out
                    a_out = z + LORA_A_GATE_TANH_RESIDUAL * torch.tanh(z)
                temp.append(self.lora_B[i](a_out) * self.scaling[i])

        if temp:
            delta = torch.cat(temp, dim=1)
            if self.current_b_gate is not None:
                b_gate = self.current_b_gate.to(device=delta.device, dtype=delta.dtype)
                delta = delta * b_gate
            result = result + delta

        return result

    def merge(self):
        """Merge LoRA weights into the original conv weight."""
        # Grouped convolutions (e.g. depthwise) have weight shape [out, 1, kH, kW],
        # which is incompatible with the LoRA delta shape. Skip merge for these;
        # the forward pass still applies LoRA correctly.
        if self.conv.groups > 1:
            return

        if self.conv.weight.size(2) == 1 and self.conv.weight.size(3) == 1:
            # 1x1 conv
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
            # 3x3 or larger conv
            full_in = self.conv.in_channels
            full_out = self.conv.out_channels
            kh, kw = self.conv.weight.size(2), self.conv.weight.size(3)
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
        """Return the full delta weight tensor (for saving/analysis)."""
        full_in = self.conv.in_channels
        full_out = self.conv.out_channels
        kh, kw = self.conv.weight.size(2), self.conv.weight.size(3)
        full_delta = torch.zeros(
            full_out, full_in, kh, kw,
            device=self.conv.weight.device, dtype=self.conv.weight.dtype
        )
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

    Args:
        model: the PyTorch model.
        r: list of ranks per hierarchy level, e.g. [2, 4, 6, 8].
        lora_alpha: list of alpha values, same length as r.
        lora_dropout: dropout rate for LoRA paths.
        target_module_names: if set, only replace modules whose full name
                             CONTAINS any of these strings.
        exclude_module_names: if set, skip modules whose full name
                              CONTAINS any of these strings.
        verbose: print replaced modules.
    Returns:
        model with MELoRA applied (modified in-place).
    """
    exclude_names = exclude_module_names or []

    n_total = 0
    n_replaced = 0
    n_skipped = 0

    for name, module in list(model.named_modules()):
        n_total += 1

        if not isinstance(module, nn.Conv2d):
            continue

        # skip 1x1 pointwise in lora_B path (or any lora_ named module)
        if any(s in name for s in exclude_names) or "lora_" in name:
            continue

        # filter by target names
        if target_module_names is not None:
            if not any(tgt in name for tgt in target_module_names):
                continue

        # skip if channels not divisible by l_num. For single-channel input
        # layers, fall back to a single branch whose rank matches the
        # MELoRA-equivalent ConvLoRA rank, i.e. the average rank across
        # branches, so parameter count stays fair to the multi-branch setting.
        l_num = len(r)
        effective_r = r
        effective_alpha = lora_alpha
        if module.in_channels % l_num != 0 or module.out_channels % l_num != 0:
            can_collapse = (
                module.in_channels < l_num
                and module.out_channels % 1 == 0
                and len(r) > 1
            )
            if can_collapse:
                effective_r = [sum(r) // len(r)]
                effective_alpha = [sum(lora_alpha) // len(lora_alpha)]
                if verbose:
                    print(f"[MELoRA] COLLAPSE {name}: in={module.in_channels}, "
                          f"out={module.out_channels}, r={r}->{effective_r}, "
                          f"alpha={lora_alpha}->{effective_alpha}")
            else:
                if verbose:
                    print(f"[MELoRA] SKIP {name}: in={module.in_channels}, out={module.out_channels} "
                          f"not divisible by l_num={l_num}")
                n_skipped += 1
                continue

        # get parent and attribute name to replace
        parent_name = name.rsplit(".", 1)
        if len(parent_name) == 2:
            parent_path, attr = parent_name
            parent = dict(model.named_modules())[parent_path]
        else:
            # top-level module
            parent = model
            attr = name

        melora_conv = MELoRAConv2d(module, effective_r, effective_alpha, lora_dropout)
        setattr(parent, attr, melora_conv)
        n_replaced += 1

        if verbose:
            print(f"[MELoRA] REPLACE {name}: in={module.in_channels}, "
                  f"out={module.out_channels}, r={effective_r}")

    if verbose:
        print(f"[MELoRA] Summary: total nn.Conv2d={n_total}, replaced={n_replaced}, "
              f"skipped={n_skipped}")

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


def get_melora_layers(model: nn.Module) -> List[MELoRAConv2d]:
    """Return MELoRA layers in module traversal order."""
    return [module for module in model.modules() if isinstance(module, MELoRAConv2d)]


def assign_melora_gate_indices(model: nn.Module) -> int:
    """Assign each MELoRA layer stable A/B indices in the gate vector."""
    layers = get_melora_layers(model)
    for idx, layer in enumerate(layers):
        layer.a_gate_index = 2 * idx
        layer.b_gate_index = 2 * idx + 1
    return 2 * len(layers)


def set_melora_gates(model: nn.Module, gates: torch.Tensor) -> None:
    """Attach batch-shared gates [2 * n_lora] to MELoRA layers for forward."""
    if gates.dim() != 1:
        raise ValueError(f"Expected LoRA gates with shape [2 * n_lora], got {tuple(gates.shape)}")
    layers = get_melora_layers(model)
    expected_gate_dim = 2 * len(layers)
    if gates.numel() != expected_gate_dim:
        raise ValueError(
            f"Expected {expected_gate_dim} LoRA gates for {len(layers)} layers, got {gates.numel()}"
        )
    for layer in layers:
        if layer.a_gate_index is None or layer.b_gate_index is None:
            raise RuntimeError("MELoRA gate indices have not been assigned")
        layer.current_a_gate = gates[layer.a_gate_index].view(1, 1, 1, 1)
        layer.current_b_gate = gates[layer.b_gate_index].view(1, 1, 1, 1)


def clear_melora_gates(model: nn.Module) -> None:
    """Clear gates so MELoRA layers use the standard ungated forward."""
    for layer in get_melora_layers(model):
        layer.current_a_gate = None
        layer.current_b_gate = None


def get_melora_state_dict(model: nn.Module) -> dict:
    """Return state dict containing only MELoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
