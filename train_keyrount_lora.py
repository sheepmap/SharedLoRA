import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import json
import numpy as np
import argparse
import os
import re
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,SliceDisplayDataDev
from models import DnCn
from MC_DDPM_SH.models.melora_utils import (
    apply_melora_to_model,
    assign_melora_gate_indices,
    clear_melora_gates,
    set_melora_gates,
    set_melora_trainable,
)
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Keep the gate input-to-output mapping simple at initialization while
        # avoiding the gate=0, LoRA-delta=0 dead start that blocks gradients.
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


def split_ab_gate_tensor(gates):
    """Split LoRA gate tensor into A/B views with the same leading dims."""
    if gates.shape[-1] % 2 != 0:
        raise ValueError(f"Expected an even gate dimension, got {tuple(gates.shape)}")
    return gates[..., 0::2], gates[..., 1::2]


def get_ab_gate_mean_values(gates):
    """Return overall, A-only, and B-only gate means for tensor or ndarray gates."""
    if torch.is_tensor(gates):
        a_gates, b_gates = split_ab_gate_tensor(gates)
        return gates.mean().item(), a_gates.mean().item(), b_gates.mean().item()

    a_gates = gates[..., 0::2]
    b_gates = gates[..., 1::2]
    return float(np.mean(gates)), float(np.mean(a_gates)), float(np.mean(b_gates))


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


def get_gate_mode_name(use_ab_gate):
    return 'ab' if use_ab_gate else 'single'


def use_lora_ab_gate(args):
    return bool(getattr(args, 'use_lora_ab_gate', True))


def get_gate_stats(gates, use_ab_gate):
    gate_mean = gates.mean().item() if torch.is_tensor(gates) else float(np.mean(gates))
    if not use_ab_gate:
        return gate_mean, None, None
    a_gate_mean, b_gate_mean = None, None
    if torch.is_tensor(gates):
        _, a_gate_mean, b_gate_mean = get_ab_gate_mean_values(gates)
    else:
        _, a_gate_mean, b_gate_mean = get_ab_gate_mean_values(gates)
    return gate_mean, a_gate_mean, b_gate_mean


def infer_use_lora_ab_gate(args, lora_state, n_lora):
    gate_bias = lora_state.get('lora_gate_net.mlp.4.bias')
    if gate_bias is None:
        return use_lora_ab_gate(args)
    actual_gate_dim = validate_lora_gate_state(lora_state, [n_lora, 2 * n_lora])
    inferred_use_ab_gate = infer_lora_ab_gate_from_dim(actual_gate_dim, n_lora)
    setattr(args, 'use_lora_ab_gate', inferred_use_ab_gate)
    return inferred_use_ab_gate


def get_batch_acceleration_value(acc_idx, acc_factors):
    if torch.is_tensor(acc_idx):
        acc_values = acc_idx.detach().view(-1).cpu().tolist()
    elif isinstance(acc_idx, (list, tuple)):
        acc_values = list(acc_idx)
    else:
        acc_values = [acc_idx]

    acc_indices = []
    for value in acc_values:
        if torch.is_tensor(value):
            value = value.item()
        acc_indices.append(int(value))

    if not acc_indices:
        raise ValueError("Empty acc_idx batch is not allowed")
    if len(set(acc_indices)) != 1:
        raise RuntimeError(f"Expected a batch with one acceleration factor, got indices {acc_indices}")

    return parse_acceleration_factor(acc_factors[acc_indices[0]])


def forward_with_lora_gates(model, us_input, input_kspace, mask, acc_idx, acc_factors):
    inner = _inner_model(model)
    if not hasattr(inner, 'lora_gate_net'):
        return model(us_input, input_kspace, mask), None

    acc_value = get_batch_acceleration_value(acc_idx, acc_factors)
    gate_input = torch.tensor([acc_value], dtype=us_input.dtype, device=us_input.device)
    gates = inner.lora_gate_net(gate_input)
    set_melora_gates(model, gates)
    try:
        output = model(us_input, input_kspace, mask)
    finally:
        clear_melora_gates(model)
    return output, gates


def forward_with_grouped_lora_gates(model, us_input, input_kspace, mask, acc_idx, acc_factors):
    """Handle mixed-acc batches by running one forward per acceleration factor."""
    inner = _inner_model(model)
    if not hasattr(inner, 'lora_gate_net'):
        return model(us_input, input_kspace, mask), None

    acc_tensor = acc_idx if torch.is_tensor(acc_idx) else torch.as_tensor(acc_idx)
    acc_tensor = acc_tensor.to(us_input.device).view(-1)
    unique_acc = torch.unique(acc_tensor)
    if unique_acc.numel() <= 1:
        return forward_with_lora_gates(model, us_input, input_kspace, mask, acc_tensor, acc_factors)

    outputs = None
    gate_matrix = None
    for acc_value in unique_acc.tolist():
        selector = acc_tensor == acc_value
        chunk_output, chunk_gates = forward_with_lora_gates(
            model,
            us_input[selector],
            input_kspace[selector],
            mask[selector],
            acc_tensor[selector],
            acc_factors,
        )
        if outputs is None:
            outputs = chunk_output.new_empty((us_input.shape[0],) + tuple(chunk_output.shape[1:]))
        outputs[selector] = chunk_output

        if chunk_gates is not None:
            if gate_matrix is None:
                gate_matrix = chunk_gates.new_empty((us_input.shape[0], chunk_gates.numel()))
            gate_matrix[selector] = chunk_gates.unsqueeze(0).expand(int(selector.sum().item()), -1)

    return outputs, gate_matrix


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

def create_datasets(args):


    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')

    train_data = SliceData(
        args.train_path, acc_factors, dataset_types, mask_types, 'train', args.usmask_path,
        data_acceleration_factor=args.data_acceleration_factor,
        expand_acc_factors=False,
    )
    dev_data = SliceData(
        args.validation_path, acc_factors, dataset_types, mask_types, 'validation', args.usmask_path,
        data_acceleration_factor=args.data_acceleration_factor,
        expand_acc_factors=True,
        return_metadata=True,
    )

    display1_data = dev_data

    return dev_data, train_data, display1_data

def create_data_loaders(args):
    dev_data, train_data, display1_data = create_datasets(args)

    display_step = max(1, len(display1_data) // 16)
    display1 = [display1_data[i] for i in range(0, len(display1_data), display_step)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=0 (原始, 单进程加载)
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        # num_workers=0 (原始, 单进程加载)
        num_workers=4,
        pin_memory=True
    )
    display_loader1 = DataLoader(
        dataset=display1,
        batch_size=16,
        shuffle=True,
        # num_workers=0 (原始, 单进程加载)
        num_workers=4,
        pin_memory=True
    )

    return train_loader, dev_loader, display_loader1


def build_mask_bank(acc_factors, mask_types, dataset_types, usmask_path, device):
    """Load pre-generated fixed masks from .npy files onto GPU."""
    mask_bank = {}
    for ds in dataset_types:
        for mt in mask_types:
            for af in acc_factors:
                path = os.path.join(usmask_path, ds, mt, f'mask_{af}.npy')
                mask_bank[(ds, mt, af)] = torch.from_numpy(np.load(path)).to(device)
    return mask_bank


def gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types):
    """Perform FFT, masking, and IFFT on GPU using pre-loaded fixed masks."""
    B = target.shape[0]
    masks = []
    for i in range(B):
        di = ds_idx[i].item() if hasattr(ds_idx[i], 'item') else ds_idx[i]
        mi = mask_idx[i].item() if hasattr(mask_idx[i], 'item') else mask_idx[i]
        ai = acc_idx[i].item() if hasattr(acc_idx[i], 'item') else acc_idx[i]
        key = (dataset_types[di], mask_types[mi], acc_factors[ai])
        masks.append(mask_bank[key])
    mask = torch.stack(masks)                                          # (B, H, W)
    kspace = torch.fft.fft2(target, norm='ortho')                     # GPU FFT
    us_kspace = kspace * mask.unsqueeze(1)
    us_img = torch.abs(torch.fft.ifft2(us_kspace, norm='ortho'))      # GPU IFFT
    return us_img.unsqueeze(1), torch.view_as_real(us_kspace), mask


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, writer,
                mask_bank, acc_factors, mask_types, dataset_types):

    model.train()
    avg_loss_by_acc = {}
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        target, acc_idx, mask_idx, ds_idx = data
        target = target.unsqueeze(1).to(args.device)
        batch_acc_idx = random.randrange(len(acc_factors))
        acc_idx = torch.full_like(acc_idx, batch_acc_idx)
        batch_acc_factor = acc_factors[batch_acc_idx]

        us_input_high, input_kspace_high, mask_high = gpu_undersample(
            target, acc_idx, mask_idx, ds_idx, mask_bank,
            acc_factors, mask_types, dataset_types)

        us_input_high = us_input_high.squeeze(1).float()
        input_kspace_high = input_kspace_high.squeeze(1).float()
        target = target.float()

        optimizer.zero_grad()
        output, gates = forward_with_lora_gates(
            model, us_input_high, input_kspace_high, mask_high, acc_idx, acc_factors
        )
        loss = F.l1_loss(output, target)
        loss.backward()
        del output

        optimizer.step()
        scheduler.step()

        prev_avg = avg_loss_by_acc.get(batch_acc_factor)
        avg_loss_by_acc[batch_acc_factor] = (
            0.99 * prev_avg + 0.01 * loss.item() if prev_avg is not None else loss.item()
        )
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)
        if gates is not None:
            gate_mean, a_gate_mean, b_gate_mean = get_gate_stats(
                gates.detach(), use_lora_ab_gate(args)
            )
            writer.add_scalar('TrainScaleMean', gate_mean, global_step + iter)
            writer.add_scalar(f'TrainScaleMean/{batch_acc_factor}', gate_mean, global_step + iter)
            if a_gate_mean is not None:
                writer.add_scalar('TrainAScaleMean', a_gate_mean, global_step + iter)
                writer.add_scalar(f'TrainAScaleMean/{batch_acc_factor}', a_gate_mean, global_step + iter)
            if b_gate_mean is not None:
                writer.add_scalar('TrainBScaleMean', b_gate_mean, global_step + iter)
                writer.add_scalar(f'TrainBScaleMean/{batch_acc_factor}', b_gate_mean, global_step + iter)
        writer.add_scalar(f'TrainLoss/{batch_acc_factor}', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            gate_log = ''
            if gates is not None:
                gate_mean, a_gate_mean, b_gate_mean = get_gate_stats(
                    gates.detach(), use_lora_ab_gate(args)
                )
                gate_values = json.dumps(
                    [round(v, 4) for v in gates.detach().cpu().tolist()],
                    ensure_ascii=True,
                )
                gate_log = f' ScaleMean[{batch_acc_factor}] = {gate_mean:.4g} '
                if a_gate_mean is not None:
                    gate_log += f'AScaleMean[{batch_acc_factor}] = {a_gate_mean:.4g} '
                if b_gate_mean is not None:
                    gate_log += f'BScaleMean[{batch_acc_factor}] = {b_gate_mean:.4g} '
                gate_log += f'Scales[{batch_acc_factor}] = {gate_values} '
            avg_loss_log = ' '.join(
                f'AvgLoss[{acc}] = {avg_loss_by_acc[acc]:.4g}'
                for acc in acc_factors if acc in avg_loss_by_acc
            )
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'L1 Loss = {loss.item():.4g} Acc = {batch_acc_factor} '
                f'{avg_loss_log} '
                f'{gate_log}'
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    mean_avg_loss = np.mean(list(avg_loss_by_acc.values())) if avg_loss_by_acc else 0.
    return mean_avg_loss, time.perf_counter() - start_epoch

def evaluate(args, epoch, model, data_loader, writer, mask_bank, acc_factors, mask_types, dataset_types):

    model.eval()
    losses = []
    psnr_list = []
    ssim_list = []
    gate_means_by_acc = {acc: [] for acc in acc_factors}
    use_ab_gate = use_lora_ab_gate(args)
    a_gate_means_by_acc = {acc: [] for acc in acc_factors} if use_ab_gate else None
    b_gate_means_by_acc = {acc: [] for acc in acc_factors} if use_ab_gate else None
    gate_vectors_by_acc = {acc: [] for acc in acc_factors}
    losses_by_acc = {acc: [] for acc in acc_factors}
    psnr_by_acc = {acc: [] for acc in acc_factors}
    ssim_by_acc = {acc: [] for acc in acc_factors}
    volume_predictions = {}
    volume_targets = {}
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            target, acc_idx, mask_idx, ds_idx, fnames, slices = data
            target = target.unsqueeze(1).to(args.device)
            us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types)

            us_input = us_input.squeeze(1).float()  # [B, 1, 1, H, W] -> [B, 1, H, W]
            input_kspace = input_kspace.squeeze(1).float()  # [B, 1, H, W, 2] -> [B, H, W, 2]
            target = target.float()

            output, gates = forward_with_grouped_lora_gates(
                model, us_input, input_kspace, mask, acc_idx, acc_factors
            )

            loss = F.mse_loss(output,target)
            losses.append(loss.item())

            output_np = output.detach().cpu().numpy().squeeze(1)  # [B, H, W]
            target_np = target.detach().cpu().numpy().squeeze(1)
            gate_vectors = None
            gate_means = None
            a_gate_means = None
            b_gate_means = None
            if gates is not None:
                gates_cpu = gates.detach().cpu()
                if gates_cpu.dim() == 1:
                    gate_vectors = gates_cpu.unsqueeze(0).expand(output_np.shape[0], -1).numpy()
                else:
                    gate_vectors = gates_cpu.numpy()
                gate_means = gate_vectors.mean(axis=1)
                if use_ab_gate:
                    a_gate_means = gate_vectors[:, 0::2].mean(axis=1)
                    b_gate_means = gate_vectors[:, 1::2].mean(axis=1)
            for b in range(output_np.shape[0]):
                acc_factor = acc_factors[acc_idx[b].item() if hasattr(acc_idx[b], 'item') else acc_idx[b]]
                losses_by_acc[acc_factor].append(float(np.mean((output_np[b] - target_np[b]) ** 2)))
                if gate_vectors is not None:
                    gate_means_by_acc[acc_factor].append(float(gate_means[b]))
                    if use_ab_gate:
                        a_gate_means_by_acc[acc_factor].append(float(a_gate_means[b]))
                        b_gate_means_by_acc[acc_factor].append(float(b_gate_means[b]))
                    gate_vectors_by_acc[acc_factor].append(gate_vectors[b])
                volume_key = (fnames[b], acc_factor)
                volume_predictions.setdefault(volume_key, []).append((int(slices[b]), output_np[b]))
                volume_targets.setdefault(volume_key, []).append((int(slices[b]), target_np[b]))

        avg_loss = np.mean(losses)
        for volume_key in sorted(volume_predictions.keys()):
            pred_slices = np.stack([pred for _, pred in sorted(volume_predictions[volume_key], key=lambda x: x[0])], axis=0)
            gt_slices = np.stack([gt for _, gt in sorted(volume_targets[volume_key], key=lambda x: x[0])], axis=0)
            gt_volume = np.transpose(gt_slices, (1, 2, 0))
            pred_volume = np.transpose(pred_slices, (1, 2, 0))
            data_range = gt_volume.max()
            if data_range > 0:
                volume_psnr = peak_signal_noise_ratio(gt_volume, pred_volume, data_range=data_range)
                psnr_list.append(volume_psnr)
                ssim_per_slice = [
                    structural_similarity(gt_slices[i], pred_slices[i], data_range=gt_slices[i].max())
                    for i in range(gt_slices.shape[0]) if gt_slices[i].max() > 0
                ]
                if ssim_per_slice:
                    volume_ssim = float(np.mean(ssim_per_slice))
                    ssim_list.append(volume_ssim)
                else:
                    volume_ssim = None
                acc_factor = volume_key[1]
                psnr_by_acc[acc_factor].append(volume_psnr)
                if volume_ssim is not None:
                    ssim_by_acc[acc_factor].append(volume_ssim)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        writer.add_scalar('Dev_Loss', avg_loss, epoch)
        writer.add_scalar('Dev_PSNR', avg_psnr, epoch)
        writer.add_scalar('Dev_SSIM', avg_ssim, epoch)
        for acc_factor in acc_factors:
            if losses_by_acc[acc_factor]:
                acc_loss_mean = float(np.mean(losses_by_acc[acc_factor]))
                writer.add_scalar(f'Dev_Loss/{acc_factor}', acc_loss_mean, epoch)
            if psnr_by_acc[acc_factor]:
                acc_psnr_mean = float(np.mean(psnr_by_acc[acc_factor]))
                writer.add_scalar(f'Dev_PSNR/{acc_factor}', acc_psnr_mean, epoch)
            if ssim_by_acc[acc_factor]:
                acc_ssim_mean = float(np.mean(ssim_by_acc[acc_factor]))
                writer.add_scalar(f'Dev_SSIM/{acc_factor}', acc_ssim_mean, epoch)
            if gate_means_by_acc[acc_factor]:
                acc_gate_mean = float(np.mean(gate_means_by_acc[acc_factor]))
                writer.add_scalar(f'Dev_ScaleMean/{acc_factor}', acc_gate_mean, epoch)
            if use_ab_gate and a_gate_means_by_acc[acc_factor]:
                writer.add_scalar(f'Dev_AScaleMean/{acc_factor}', float(np.mean(a_gate_means_by_acc[acc_factor])), epoch)
            if use_ab_gate and b_gate_means_by_acc[acc_factor]:
                writer.add_scalar(f'Dev_BScaleMean/{acc_factor}', float(np.mean(b_gate_means_by_acc[acc_factor])), epoch)
        eval_log = format_eval_log(
            epoch, args.num_epochs, acc_factors, losses_by_acc, psnr_by_acc, ssim_by_acc,
            gate_means_by_acc, a_gate_means_by_acc, b_gate_means_by_acc, gate_vectors_by_acc, use_ab_gate,
        )
        logging.info(eval_log)

    metrics_by_acc = {}
    for acc_factor in acc_factors:
        metrics_by_acc[acc_factor] = {}
        if losses_by_acc[acc_factor]:
            metrics_by_acc[acc_factor]['loss'] = float(np.mean(losses_by_acc[acc_factor]))
        if psnr_by_acc[acc_factor]:
            metrics_by_acc[acc_factor]['psnr'] = float(np.mean(psnr_by_acc[acc_factor]))
        if ssim_by_acc[acc_factor]:
            metrics_by_acc[acc_factor]['ssim'] = float(np.mean(ssim_by_acc[acc_factor]))

    return avg_loss, avg_psnr, avg_ssim, time.perf_counter() - start, metrics_by_acc, eval_log


def visualize(args, epoch, model, data_loader, writer, datasettype_string, mask_bank=None, acc_factors=None, mask_types=None, dataset_types=None):


    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            if mask_bank is not None:
                # SliceData returns (target, acc_idx, mask_idx, ds_idx)
                if len(data) == 6:
                    target, acc_idx, mask_idx, ds_idx, _, _ = data
                else:
                    target, acc_idx, mask_idx, ds_idx = data
                target = target.unsqueeze(1).to(args.device)
                us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types)
                us_input = us_input.squeeze(1)  # [B, 1, 1, H, W] -> [B, 1, H, W]
                input_kspace = input_kspace.squeeze(1)  # [B, 1, H, W, 2] -> [B, H, W, 2]
            else:
                # SliceDisplayDataDev returns (input_img, input_kspace, target, mask)
                us_input, input_kspace, target, mask = data
                us_input = us_input.unsqueeze(1).to(args.device)
                input_kspace = input_kspace.to(args.device)
                target = target.unsqueeze(1).to(args.device)
                mask = mask.to(args.device)

            us_input = us_input.float()
            target = target.float()
            if mask_bank is not None:
                output, _ = forward_with_grouped_lora_gates(
                    model, us_input, input_kspace, mask, acc_idx, acc_factors
                )
            else:
                output = model(us_input, input_kspace, mask)

            save_image(us_input, 'Input_{}'.format(datasettype_string))
            save_image(target, 'Target_{}'.format(datasettype_string))
            save_image(output, 'Reconstruction_{}'.format(datasettype_string))
            save_image(torch.abs(target.float() - output.float()), 'Error_{}'.format(datasettype_string))
            break

def _make_melora_dirname(args):
    """Build subdirectory name from MELORA_R and MELORA_TARGET."""
    r_str = args.melora_r.replace(',', '_')
    target_str = args.melora_target.replace(',', '_') if args.melora_target else 'all'
    gate_mode = get_gate_mode_name(use_lora_ab_gate(args))
    return f'r{r_str}_{target_str}_gate_{gate_mode}'


def use_lora_gate_net(args):
    return bool(getattr(args, 'use_lora_gate_net', False))


def format_eval_log(epoch, num_epochs, acc_factors, losses_by_acc, psnr_by_acc, ssim_by_acc,
                    gate_means_by_acc, a_gate_means_by_acc, b_gate_means_by_acc, gate_vectors_by_acc,
                    use_ab_gate):
    metric_logs = []
    gate_logs = []
    for acc_factor in acc_factors:
        if losses_by_acc[acc_factor]:
            acc_loss_mean = float(np.mean(losses_by_acc[acc_factor]))
            metric_logs.append(f'DevLoss[{acc_factor}] = {acc_loss_mean:.4g}')
        if psnr_by_acc[acc_factor]:
            acc_psnr_mean = float(np.mean(psnr_by_acc[acc_factor]))
            metric_logs.append(f'PSNR[{acc_factor}] = {acc_psnr_mean:.4g}')
        if ssim_by_acc[acc_factor]:
            acc_ssim_mean = float(np.mean(ssim_by_acc[acc_factor]))
            metric_logs.append(f'SSIM[{acc_factor}] = {acc_ssim_mean:.4g}')
        if gate_means_by_acc[acc_factor]:
            acc_gate_mean = float(np.mean(gate_means_by_acc[acc_factor]))
            if gate_vectors_by_acc[acc_factor]:
                gate_vector_mean = np.mean(np.stack(gate_vectors_by_acc[acc_factor], axis=0), axis=0)
                gate_vector_str = json.dumps(
                    [round(float(v), 4) for v in gate_vector_mean.tolist()],
                    ensure_ascii=True,
                )
                gate_log = f'ScaleMean[{acc_factor}] = {acc_gate_mean:.4g} '
                if use_ab_gate:
                    acc_a_gate_mean = float(np.mean(a_gate_means_by_acc[acc_factor]))
                    acc_b_gate_mean = float(np.mean(b_gate_means_by_acc[acc_factor]))
                    gate_log += (
                        f'AScaleMean[{acc_factor}] = {acc_a_gate_mean:.4g} '
                        f'BScaleMean[{acc_factor}] = {acc_b_gate_mean:.4g} '
                    )
                gate_log += f'ScaleVecMean[{acc_factor}] = {gate_vector_str}'
                gate_logs.append(gate_log)
            else:
                gate_log = f'ScaleMean[{acc_factor}] = {acc_gate_mean:.4g}'
                if use_ab_gate:
                    acc_a_gate_mean = float(np.mean(a_gate_means_by_acc[acc_factor]))
                    acc_b_gate_mean = float(np.mean(b_gate_means_by_acc[acc_factor]))
                    gate_log += (
                        f' AScaleMean[{acc_factor}] = {acc_a_gate_mean:.4g}'
                        f' BScaleMean[{acc_factor}] = {acc_b_gate_mean:.4g}'
                    )
                gate_logs.append(gate_log)

    parts = metric_logs + gate_logs
    prefix = f'Eval Epoch = [{epoch:3d}/{num_epochs:3d}]'
    return prefix if not parts else f'{prefix} ' + ' '.join(parts)


def save_epoch_validation_log(save_dir, epoch, eval_log, summary_log):
    log_dir = save_dir / 'validation_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'epoch_{epoch:04d}.txt'
    log_path.write_text(f'{eval_log}\n{summary_log}\n', encoding='utf-8')

def save_model(args, save_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best):
    """Save LoRA adapter, optional gate net, and training metadata. No base weights."""
    lora_state = {k: v for k, v in model.state_dict().items()
                  if 'lora_' in k or 'lora_gate_net' in k}

    # Checkpoint: training metadata only, for resume
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        },
        f=save_dir / 'checkpoint.pt'
    )

    # Adapter: LoRA and gate-net weights only
    torch.save(lora_state, f=save_dir / 'adapter.pt')

    if is_new_best:
        shutil.copyfile(save_dir / 'adapter.pt', save_dir / 'adapter_best.pt')


def build_model_from_pretrained(args):
    """Load pretrained base weights, apply MELoRA, freeze base, return model with only lora trainable."""
    # Load pretrained base model
    pretrained = load_torch_checkpoint(args.pretrained_checkpoint)
    base_state = pretrained['model']

    # Create model and load base weights
    model = DnCn(args, n_channels=1).to(args.device)
    model.load_state_dict(base_state, strict=True)

    # Apply MELoRA
    melora_r = [int(x.strip()) for x in args.melora_r.split(",")]
    melora_alpha = [int(x.strip()) for x in args.melora_alpha.split(",")]
    assert len(melora_r) == len(melora_alpha), \
        f"melora_r len ({len(melora_r)}) != melora_alpha len ({len(melora_alpha)})"
    target = [x.strip() for x in args.melora_target.split(",")] if args.melora_target else None
    logger.info(f"applying MELoRA Conv2d: r={melora_r}, alpha={melora_alpha}, target={target}")
    apply_melora_to_model(model, melora_r, melora_alpha,
                          lora_dropout=args.melora_dropout,
                          target_module_names=target,
                          verbose=True)

    # Freeze base, only lora trainable
    set_melora_trainable(model)

    if use_lora_gate_net(args):
        gate_dim = assign_melora_gate_indices(model, use_lora_ab_gate=use_lora_ab_gate(args))
        model.lora_gate_net = LoRAGateNet(gate_dim).to(args.device)
        for param in model.lora_gate_net.parameters():
            param.requires_grad = True
        n_lora = len([module for module in model.modules() if 'MELoRAConv2d' in type(module).__name__])
        logger.info(
            f"LoRA gate net enabled: gate_mode={get_gate_mode_name(use_lora_ab_gate(args))}, "
            f"n_lora={n_lora}, gate_dim={gate_dim}"
        )
    else:
        logger.info("LoRA gate net disabled: using standard ConvLoRA training")

    # Verify and report which params are trainable
    total_count = sum(p.numel() for p in model.parameters())
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = total_count - frozen_count
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_sample = [n for n, p in model.named_parameters() if not p.requires_grad][:5]

    logger.info(f"{'='*60}")
    logger.info(f"MELoRA verification:")
    logger.info(f"  Total     params: {total_count:,}")
    logger.info(f"  Frozen    params: {frozen_count:,} ({frozen_count/total_count*100:.1f}%)")
    logger.info(f"  Trainable params: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    logger.info(f"  Trainable layers ({len(trainable_names)}):")
    for n in trainable_names[:10]:
        logger.info(f"    鉁?{n}")
    if len(trainable_names) > 10:
        logger.info(f"    ... and {len(trainable_names) - 10} more")
    logger.info(f"  Frozen layer sample:")
    for n in frozen_sample:
        logger.info(f"    鉁?{n}")
    logger.info(f"{'='*60}")

    return model


def load_model(checkpoint_file):
    """Resume from a LoRA checkpoint.pt (metadata only, no model weights)."""
    checkpoint = load_torch_checkpoint(checkpoint_file)
    args = checkpoint['args']

    # Load base weights from pretrained checkpoint
    pretrained = load_torch_checkpoint(args.pretrained_checkpoint)
    base_state = pretrained['model']

    model = DnCn(args, n_channels=1).to(args.device)
    model.load_state_dict(base_state, strict=True)

    # Apply MELoRA
    melora_r = [int(x.strip()) for x in args.melora_r.split(",")]
    melora_alpha = [int(x.strip()) for x in args.melora_alpha.split(",")]
    target = [x.strip() for x in args.melora_target.split(",")] if args.melora_target else None
    apply_melora_to_model(model, melora_r, melora_alpha,
                          lora_dropout=args.melora_dropout,
                          target_module_names=target,
                          verbose=True)

    adapter_path = pathlib.Path(checkpoint_file).parent / 'adapter.pt'
    lora_state = load_torch_checkpoint(adapter_path)
    has_gate_net_weights = any(k.startswith('lora_gate_net.') for k in lora_state.keys())

    set_melora_trainable(model)
    if use_lora_gate_net(args) or has_gate_net_weights:
        n_lora = len([module for module in model.modules() if 'MELoRAConv2d' in type(module).__name__])
        inferred_use_ab_gate = infer_use_lora_ab_gate(args, lora_state, n_lora)
        gate_dim = assign_melora_gate_indices(model, use_lora_ab_gate=inferred_use_ab_gate)
        model.lora_gate_net = LoRAGateNet(gate_dim).to(args.device)
        for param in model.lora_gate_net.parameters():
            param.requires_grad = True

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # Load lora adapter (same directory as checkpoint.pt)
    model.load_state_dict(lora_state, strict=False)

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.AdamW(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        runtime_batch_size = args.batch_size
        runtime_num_epochs = args.num_epochs
        runtime_device = args.device
        runtime_exp_dir = args.exp_dir
        runtime_report_interval = args.report_interval
        runtime_data_parallel = args.data_parallel
        runtime_checkpoint = args.checkpoint
        args = checkpoint['args']
        # Keep selected runtime overrides so resume can extend training cleanly.
        args.batch_size = runtime_batch_size
        args.num_epochs = runtime_num_epochs
        args.device = runtime_device
        args.exp_dir = runtime_exp_dir
        args.report_interval = runtime_report_interval
        args.data_parallel = runtime_data_parallel
        args.resume = True
        args.checkpoint = runtime_checkpoint
        best_psnr = checkpoint.get('best_psnr', 0.)
        start_epoch = checkpoint['epoch'] + 1
        if args.num_epochs <= start_epoch:
            raise ValueError(
                f"Resumed checkpoint is already at epoch {start_epoch - 1}, "
                f"so --num-epochs must be greater than {start_epoch}. Got {args.num_epochs}."
            )
        del checkpoint
    else:
        model = build_model_from_pretrained(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_psnr = 0.
        start_epoch = 0

    melora_dir = args.exp_dir / 'melora' / _make_melora_dirname(args)
    melora_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(melora_dir / 'summary'))

    logging.info(args)
    logging.info(model)
    train_loader, dev_loader, display1_loader = create_data_loaders(args)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.05
    )
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    mask_bank = build_mask_bank(acc_factors, mask_types, dataset_types, args.usmask_path, args.device)

    for epoch in range(start_epoch, args.num_epochs):

        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, scheduler, writer,
            mask_bank, acc_factors, mask_types, dataset_types)
        dev_loss, dev_psnr, dev_ssim, dev_time, dev_metrics_by_acc, eval_log = evaluate(
            args, epoch, model, dev_loader, writer, mask_bank, acc_factors, mask_types, dataset_types
        )
        visualize(
            args, epoch, model, display1_loader, writer, 't1',
            mask_bank, acc_factors, mask_types, dataset_types
        )

        is_new_best = dev_psnr > best_psnr
        best_psnr = max(best_psnr, dev_psnr)
        save_model(args, melora_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best)
        dev_metric_log = ' '.join(
            ' '.join(
                f'{metric.upper()}[{acc}] = {value:.4g}'
                for metric, value in dev_metrics_by_acc[acc].items()
            )
            for acc in acc_factors if dev_metrics_by_acc.get(acc)
        )
        summary_log = (
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} PSNR = {dev_psnr:.4g} SSIM = {dev_ssim:.4g} '
            f'{dev_metric_log} '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s'
        )
        logging.info(summary_log)
        save_epoch_validation_log(melora_dir, epoch, eval_log, summary_log)
    writer.close()

def create_arg_parser():

    parser = argparse.ArgumentParser(description='LoRA fine-tuning for MR recon U-Net')
    parser.add_argument('--pretrained-checkpoint', type=pathlib.Path, required=True,
                        help='Path to pretrained base model checkpoint')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--data_acceleration_factor', type=str, default=None,
                        help='acc factor directory to read h5 from, e.g. 16x; if unset, follows --acceleration_factor')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')

    # MELoRA settings
    parser.add_argument('--melora_r', type=str, default='2,4,6,8',
                        help='comma-separated ranks per hierarchy level')
    parser.add_argument('--melora_alpha', type=str, default='16,16,16,16',
                        help='comma-separated alpha values (same length as melora_r)')
    parser.add_argument('--melora_dropout', type=float, default=0.0,
                        help='dropout rate for MELoRA paths')
    parser.add_argument('--melora_target', type=str, default='',
                        help='comma-separated name substrings to target (e.g. "down_sample_layers")')
    parser.add_argument('--use-lora-gate-net', action='store_true',
                        help='Enable LoRA gate net conditioning; if unset, use standard ConvLoRA training')
    parser.add_argument('--use-lora-ab-gate', dest='use_lora_ab_gate', action='store_true', default=True,
                        help='Use separate A/B gates per LoRA layer when gate net is enabled')
    parser.add_argument('--no-lora-ab-gate', dest='use_lora_ab_gate', action='store_false',
                        help='Use one gate per LoRA layer output when gate net is enabled')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
