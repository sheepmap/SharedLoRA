"""Convolutional PEFT adapters used by the SHFormer reconstruction models.

The adapters in this module operate on ``nn.Conv2d`` with ``groups == 1``.
They keep the original convolution frozen and replace it in-place, matching the
injection style used by :mod:`melora_utils`.
"""
import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDoRA(nn.Module):
    """DoRA adapter for a regular 2-D convolution.

    The low-rank branch updates the direction of the convolution kernel while
    ``dora_magnitude`` learns one magnitude value per output channel.  The
    magnitude is initialized from the frozen kernel norm, therefore a freshly
    inserted adapter is functionally equivalent to the original convolution.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        _require_regular_conv(conv)
        if rank <= 0:
            raise ValueError(f"DoRA rank must be positive, got {rank}")

        self.conv = conv
        _freeze_module(self.conv)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank

        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Conv2d(
            conv.in_channels,
            self.rank,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            groups=1,
            bias=False,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.lora_B = nn.Conv2d(
            self.rank,
            conv.out_channels,
            kernel_size=1,
            bias=False,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        kernel_norm = conv.weight.detach().flatten(1).norm(dim=1)
        self.dora_magnitude = nn.Parameter(kernel_norm.clone())
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _delta_weight(self) -> torch.Tensor:
        return torch.einsum(
            "or,rihw->oihw",
            self.lora_B.weight.squeeze(-1).squeeze(-1),
            self.lora_A.weight,
        ) * self.scaling

    def adapted_weight(self) -> torch.Tensor:
        direction = self.conv.weight + self._delta_weight()
        direction_norm = direction.flatten(1).norm(dim=1).clamp_min(torch.finfo(direction.dtype).eps)
        return direction * (self.dora_magnitude / direction_norm).view(-1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is the standard efficient DoRA form: the low-rank branch is
        # evaluated as two convolutions and the magnitude/direction ratio is
        # applied per output channel.  At evaluation (where dropout is an
        # identity) it is equivalent to a single convolution with the
        # normalized adapted weight.
        direction = self.conv.weight + self._delta_weight()
        direction_norm = direction.flatten(1).norm(dim=1).clamp_min(torch.finfo(direction.dtype).eps)
        magnitude_scale = (self.dora_magnitude / direction_norm).view(1, -1, 1, 1)
        lora = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return (self.conv(x) + lora) * magnitude_scale

    @torch.no_grad()
    def merge(self) -> None:
        """Materialize the current DoRA weight in the frozen base convolution."""
        direction = self.conv.weight + self._delta_weight()
        direction_norm = direction.flatten(1).norm(dim=1).clamp_min(torch.finfo(direction.dtype).eps)
        self.conv.weight.copy_(direction * (self.dora_magnitude / direction_norm).view(-1, 1, 1, 1))
        if self.conv.bias is not None:
            self.conv.bias.mul_(self.dora_magnitude / direction_norm)


class ConvLoRAXS(nn.Module):
    """LoRA-XS adapter for a regular 2-D convolution.

    SVD-derived bases are frozen buffers.  Only the small ``r x r``
    ``lora_xs_core`` matrix is trainable, and it is zero initialized so the
    wrapped layer initially matches the pretrained convolution exactly.
    """

    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float = 1.0):
        super().__init__()
        _require_regular_conv(conv)
        if rank <= 0:
            raise ValueError(f"LoRA-XS rank must be positive, got {rank}")

        self.conv = conv
        _freeze_module(self.conv)
        out_channels = conv.out_channels
        flat_weight = conv.weight.detach().float().reshape(out_channels, -1)
        u, singular_values, vh = torch.linalg.svd(flat_weight, full_matrices=False)
        self.rank = min(int(rank), u.shape[1])
        if self.rank < 1:
            raise ValueError(f"Unable to construct LoRA-XS basis for {tuple(conv.weight.shape)}")

        # Use the symmetric SVD factorization U sqrt(S) and sqrt(S) V^T.
        # It keeps the core update well-conditioned while preserving the
        # pretrained weight's principal low-rank subspace.
        sqrt_singular_values = singular_values[:self.rank].sqrt()
        left_basis = u[:, :self.rank] * sqrt_singular_values.unsqueeze(0)
        right_basis = sqrt_singular_values.unsqueeze(1) * vh[:self.rank, :]
        self.register_buffer("lora_xs_left_basis", left_basis.to(dtype=conv.weight.dtype))
        self.register_buffer("lora_xs_right_basis", right_basis.to(dtype=conv.weight.dtype))
        self.lora_xs_core = nn.Parameter(
            torch.zeros(self.rank, self.rank, device=conv.weight.device, dtype=conv.weight.dtype)
        )
        self.scaling = float(alpha) / self.rank

    def _delta_weight(self) -> torch.Tensor:
        delta = self.lora_xs_left_basis @ self.lora_xs_core @ self.lora_xs_right_basis
        return delta.reshape_as(self.conv.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        delta = F.conv2d(
            x,
            self._delta_weight(),
            None,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return base + delta

    @torch.no_grad()
    def merge(self) -> None:
        """Materialize the current LoRA-XS update in the frozen base convolution."""
        self.conv.weight.add_(self._delta_weight())


def _require_regular_conv(conv: nn.Conv2d) -> None:
    if conv.groups != 1:
        raise ValueError(
            "ConvDoRA and ConvLoRAXS currently support only groups=1 convolutions, "
            f"got groups={conv.groups} for {conv!r}"
        )


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _replace_conv_modules(
    model: nn.Module,
    wrapper_cls,
    wrapper_kwargs: dict,
    target_module_names: Optional[List[str]],
    exclude_module_names: Optional[List[str]],
    verbose: bool,
) -> nn.Module:
    """Replace selected regular Conv2d modules with the requested wrapper."""
    excludes = exclude_module_names or []
    replacements = []
    skipped_grouped = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if "lora_" in name or "dora_" in name:
            continue
        if any(token in name for token in excludes):
            continue
        if target_module_names is not None and not any(token in name for token in target_module_names):
            continue
        if module.groups != 1:
            skipped_grouped.append(name)
            continue
        replacements.append((name, module))

    module_map = dict(model.named_modules())
    for name, module in replacements:
        parent_path, _, attr = name.rpartition(".")
        parent = module_map[parent_path] if parent_path else model
        setattr(parent, attr, wrapper_cls(module, **wrapper_kwargs))
        if verbose:
            print(f"[ConvPEFT] REPLACE {name}: {type(module).__name__} -> {wrapper_cls.__name__}")

    if verbose:
        print(
            f"[ConvPEFT] Summary: method={wrapper_cls.__name__}, "
            f"replaced={len(replacements)}, skipped_grouped={len(skipped_grouped)}"
        )
    if not replacements:
        raise ValueError("No Conv2d layers matched the requested PEFT target modules")
    return model


def apply_dora_to_model(
    model: nn.Module,
    rank: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_module_names: Optional[List[str]] = None,
    exclude_module_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    return _replace_conv_modules(
        model,
        ConvDoRA,
        {"rank": rank, "alpha": alpha, "dropout": dropout},
        target_module_names,
        exclude_module_names,
        verbose,
    )


def apply_lora_xs_to_model(
    model: nn.Module,
    rank: int,
    alpha: float = 1.0,
    target_module_names: Optional[List[str]] = None,
    exclude_module_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    return _replace_conv_modules(
        model,
        ConvLoRAXS,
        {"rank": rank, "alpha": alpha},
        target_module_names,
        exclude_module_names,
        verbose,
    )


def apply_conv_peft(
    model: nn.Module,
    method: str,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_module_names: Optional[List[str]] = None,
    exclude_module_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Inject a non-MELoRA convolution PEFT method into ``model``."""
    method = method.lower()
    if method == "dora":
        return apply_dora_to_model(
            model, rank, alpha, dropout, target_module_names, exclude_module_names, verbose
        )
    if method == "lora-xs":
        return apply_lora_xs_to_model(
            model, rank, alpha, target_module_names, exclude_module_names, verbose
        )
    raise ValueError(f"Unsupported Conv PEFT method {method!r}; expected 'dora' or 'lora-xs'")


def set_conv_peft_trainable(model: nn.Module) -> int:
    """Freeze a model and enable only DoRA or LoRA-XS trainable parameters."""
    trainable_count = 0
    for name, parameter in model.named_parameters():
        is_adapter = "lora_" in name or "dora_" in name
        parameter.requires_grad = is_adapter
        if is_adapter:
            trainable_count += parameter.numel()
    return trainable_count


def get_conv_peft_layers(model: nn.Module) -> Iterable[nn.Module]:
    """Yield the ConvDoRA and ConvLoRAXS wrappers in traversal order."""
    return (module for module in model.modules() if isinstance(module, (ConvDoRA, ConvLoRAXS)))
