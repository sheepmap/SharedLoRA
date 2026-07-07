import pathlib
import sys
from collections import defaultdict
import argparse
import gc
import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import DnCn
import h5py
from tqdm import tqdm


class LoRAGateNet(nn.Module):
    """Predict batch-shared LoRA scales [gate_dim] from an acceleration scalar."""

    def __init__(self, n_lora):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_lora),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.ones_(self.mlp[-1].bias)

    def forward(self, x):
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() != 2:
            raise ValueError(f"Expected acceleration input with <=2 dims, got shape {tuple(x.shape)}")
        return self.mlp(x).squeeze(0)


def _inner_model(model):
    return model.module if hasattr(model, 'module') else model


def parse_acceleration_factor(acc_factor):
    match = re.search(r"[-+]?\d*\.?\d+", str(acc_factor))
    if match is None:
        raise ValueError(f"Unable to parse acceleration factor from {acc_factor!r}")
    return float(match.group(0))


def get_inference_acceleration_value(acceleration_factor):
    acc_factors = [item.strip() for item in str(acceleration_factor).split(',') if item.strip()]
    if len(acc_factors) != 1:
        raise ValueError(
            f"valid.py expects a single acceleration factor, got {acceleration_factor!r}"
        )
    return parse_acceleration_factor(acc_factors[0])


def validate_lora_gate_state(lora_state, expected_gate_dims):
    """Validate gate-net output dim against supported single-gate and A/B-gate layouts."""
    gate_bias = lora_state.get('lora_gate_net.mlp.4.bias')
    if gate_bias is None:
        return None
    actual_gate_dim = int(gate_bias.shape[0])
    if actual_gate_dim in expected_gate_dims:
        return actual_gate_dim
    expected_gate_dim_str = ', '.join(str(dim) for dim in expected_gate_dims)
    raise ValueError(
        f"LoRA gate net output dim mismatch: expected one of [{expected_gate_dim_str}], got {actual_gate_dim}. "
        "This adapter uses an unsupported gate layout for the current MELoRA layer count."
    )


def infer_lora_ab_gate_from_dim(actual_gate_dim, n_lora):
    if actual_gate_dim == 2 * n_lora:
        return True
    if actual_gate_dim == n_lora:
        return False
    raise ValueError(
        f"LoRA gate net output dim mismatch: expected {n_lora} or {2 * n_lora}, got {actual_gate_dim}."
    )

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.mask_type,args.usmask_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_torch_checkpoint(path, map_location=None):
    """
    Load trusted project checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load default `weights_only` from False to True,
    which breaks older checkpoints that store argparse.Namespace in `args`.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(checkpoint_file, use_lora=False, lora_path=None):
    checkpoint = load_torch_checkpoint(checkpoint_file)
    args = checkpoint['args']
    model = DnCn(args,n_channels=1).to(args.device)

    if use_lora and lora_path is not None:
        # LoRA inference: load base weights, apply MELoRA, load adapter, merge
        model.load_state_dict(checkpoint['model'], strict=False)

        # Load LoRA hyperparams from checkpoint.pt (saved alongside adapter)
        lora_chk = load_torch_checkpoint(pathlib.Path(lora_path).parent / 'checkpoint.pt')
        lora_args = lora_chk['args']
        melora_r = [int(x.strip()) for x in lora_args.melora_r.split(",")]
        melora_alpha = [int(x.strip()) for x in lora_args.melora_alpha.split(",")]
        target = [x.strip() for x in lora_args.melora_target.split(",")] if getattr(lora_args, 'melora_target', None) else None
        from MC_DDPM_SH.models.melora_utils import apply_melora_to_model, assign_melora_gate_indices, get_melora_layers
        apply_melora_to_model(model, melora_r, melora_alpha,
                              lora_dropout=getattr(lora_args, 'melora_dropout', 0.0),
                              target_module_names=target,
                              verbose=True)

        lora_state = load_torch_checkpoint(lora_path)
        uses_gate_net = any(k.startswith('lora_gate_net.') for k in lora_state.keys())
        if uses_gate_net:
            n_lora = len(get_melora_layers(model))
            actual_gate_dim = validate_lora_gate_state(lora_state, [n_lora, 2 * n_lora])
            inferred_use_ab_gate = infer_lora_ab_gate_from_dim(actual_gate_dim, n_lora)
            setattr(lora_args, 'use_lora_ab_gate', inferred_use_ab_gate)
            gate_dim = assign_melora_gate_indices(model, use_lora_ab_gate=inferred_use_ab_gate)
            model.lora_gate_net = LoRAGateNet(gate_dim).to(args.device)

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        model.load_state_dict(lora_state, strict=False)

        if uses_gate_net:
            gate_mode = 'ab' if getattr(lora_args, 'use_lora_ab_gate', True) else 'single'
            print(f"MELoRA adapter with {gate_mode} gate net loaded from {lora_path} for dynamic inference")
        else:
            for m in model.modules():
                if hasattr(m, 'merge'):
                    m.merge()
                    # Grouped convs skip merge (weight shape incompatible),
                    # so keep their LoRA adapters active in forward pass.
                    if m.conv.groups <= 1:
                        m.disable_adapters = True
            print(f"MELoRA adapter loaded from {lora_path} and merged for inference")
    else:
        # Original inference (no LoRA)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])

    return model


def run_unet(args, model, data_loader):
    inner = _inner_model(model)
    has_gate_net = hasattr(inner, 'lora_gate_net')
    gates = None
    if has_gate_net:
        from MC_DDPM_SH.models.melora_utils import clear_melora_gates, set_melora_gates

        acc_value = get_inference_acceleration_value(args.acceleration_factor)
        gate_input = torch.tensor([acc_value], dtype=torch.float32, device=args.device)
        gates = inner.lora_gate_net(gate_input)

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            us_input, input_kspace, target,mask,fnames,slices = data
            us_input = us_input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.to(args.device)
            mask = mask.to(args.device)

            us_input = us_input.float()

            if has_gate_net:
                set_melora_gates(model, gates)
                try:
                    recons = model(us_input,input_kspace,mask).to('cpu').squeeze(1)
                finally:
                    clear_melora_gates(model)
            else:
                recons = model(us_input,input_kspace,mask).to('cpu').squeeze(1)

            if args.dataset_type == 'cardiac':
                recons = recons[:,5:155,5:155]

            
            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

            del us_input, input_kspace, target, mask, recons, data
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    data_loader = None
    model = None
    reconstructions = None
    try:
        data_loader = create_data_loaders(args)
        model = load_model(args.checkpoint, args.use_lora, args.lora_path)
        reconstructions = run_unet(args, model, data_loader)
        save_reconstructions(reconstructions, args.out_dir)
    finally:
        if model is not None:
            model.cpu()
        del reconstructions, model, data_loader
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='If set, load a LoRA adapter and merge for inference')
    parser.add_argument('--lora_path', type=pathlib.Path, default=None,
                        help='Path to the LoRA adapter .pt file (required when --use_lora is set)')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
