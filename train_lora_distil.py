import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import cv2
import argparse
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,SliceDisplayDataDev
from models import DnCn
from MC_DDPM_SH.models.melora_utils import apply_melora_to_model, set_melora_trainable
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):


    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')

    train_data = SliceData(args.train_path,acc_factors, dataset_types,mask_types,'train', args.usmask_path)
    dev_data = SliceData(args.validation_path,acc_factors,dataset_types,mask_types,'validation', args.usmask_path)

    display_dataset_type = dataset_types[0] if dataset_types else 'mrbrain_t1'
    display_mask_type = mask_types[0] if mask_types else 'cartesian'
    display_acc_factor = acc_factors[0] if acc_factors else '4x'
    display1_data = SliceDisplayDataDev(args.validation_path, display_dataset_type, display_mask_type, display_acc_factor, args.usmask_path)

    return dev_data, train_data, display1_data

def create_data_loaders(args):
    dev_data, train_data, display1_data = create_datasets(args)

    display1 = [display1_data[i] for i in range(0, len(display1_data), len(display1_data) // 16)]

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


def gaussian_noise_aug(us_input, sigma):
    """使用 OpenCV 对 us_input 添加高斯噪声。

    将每个样本归一化到 [0, 255] 浮点范围，通过 cv2.randn 添加 N(0, sigma) 噪声，
    裁剪到 [0, 255] 防止 uint8 溢出回绕，再反归一化回原始浮点范围。

    Args:
        us_input: [B, H, W] float32 GPU 张量
        sigma: 高斯噪声标准差（uint8 尺度，如 25 或 50）

    Returns:
        与 us_input 同形状、同 dtype 的含噪张量
    """
    B, H, W = us_input.shape
    noisy = torch.empty_like(us_input)

    for i in range(B):
        sample = us_input[i].detach().cpu().numpy()
        s_min = sample.min()
        s_max = sample.max()
        data_range = s_max - s_min

        if data_range < 1e-8:
            noisy[i] = us_input[i]
            continue

        # 归一化到 [0, 255] 浮点（非 uint8，避免回绕）
        sample_255 = (sample - s_min) / data_range * 255.0

        # 使用 OpenCV 生成高斯噪声
        noise = np.zeros_like(sample_255, dtype=np.float32)
        cv2.randn(noise, 0, sigma)

        # 添加噪声并裁剪到 [0, 255]，防止溢出
        noisy_255 = np.clip(sample_255 + noise, 0.0, 255.0)

        # 反归一化回原始范围
        noisy_sample = noisy_255 / 255.0 * data_range + s_min

        noisy[i] = torch.from_numpy(noisy_sample).to(us_input.device)

    return noisy


def multi_scale_distil_loss(output, output_ref):
    """计算三个尺度(0.5x, 1x, 2x)的蒸馏L1 loss并取平均。"""
    loss_orig = F.l1_loss(output, output_ref)
    output_half = F.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True)
    ref_half = F.interpolate(output_ref, scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True)
    loss_half = F.l1_loss(output_half, ref_half)
    output_double = F.interpolate(output, scale_factor=2.0, mode='bilinear', align_corners=False)
    ref_double = F.interpolate(output_ref, scale_factor=2.0, mode='bilinear', align_corners=False)
    loss_double = F.l1_loss(output_double, ref_double)
    return (loss_orig + loss_half + loss_double) / 3


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, writer,
                mask_bank, acc_factors, mask_types, dataset_types,
                ref_model=None, mask_bank_ref=None):

    model.train()
    avg_loss = 0.
    avg_loss_img = 0.
    avg_loss_distil = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        target, acc_idx, mask_idx, ds_idx = data
        target = target.unsqueeze(1).to(args.device)

        # 高倍欠采样输入（训练分支）
        us_input_high, input_kspace_high, mask_high = gpu_undersample(
            target, acc_idx, mask_idx, ds_idx, mask_bank,
            acc_factors, mask_types, dataset_types)

        us_input_high = us_input_high.squeeze(1).float()
        input_kspace_high = input_kspace_high.squeeze(1).float()
        target = target.float()

        optimizer.zero_grad()

        # --- 蒸馏训练 ---
        if ref_model is not None and mask_bank_ref is not None and args.distil_alpha > 0:
            # 生成低倍欠采样输入（参考分支）
            ref_acc_idx = [0] * target.shape[0]
            us_input_ref, input_kspace_ref, mask_ref = gpu_undersample(
                target, ref_acc_idx, mask_idx, ds_idx, mask_bank_ref,
                args.ref_acceleration_factor.split(','),
                mask_types, dataset_types)
            us_input_ref = us_input_ref.squeeze(1).float()
            input_kspace_ref = input_kspace_ref.squeeze(1).float()

            # === Pass 1: 原始数据 ===
            output = model(us_input_high, input_kspace_high, mask_high)
            with torch.no_grad():
                output_ref = ref_model(us_input_ref, input_kspace_ref, mask_ref)
            loss_img = F.l1_loss(output, target)
            loss_distil = multi_scale_distil_loss(output, output_ref) if args.distil_multi_scale else F.l1_loss(output, output_ref)
            loss = 0.5 * (loss_img + args.distil_alpha * loss_distil) if args.aug_flip else (loss_img + args.distil_alpha * loss_distil)
            loss.backward()
            del output, output_ref

            # === Pass 2: 轻度噪声增广 (仅当 args.aug_flip=True) ===
            if args.aug_flip:
                sigma_25 = random.uniform(15, 35)
                us_noisy_25 = gaussian_noise_aug(us_input_high, sigma=sigma_25)
                output_n25 = model(us_noisy_25, input_kspace_high, mask_high)
                with torch.no_grad():
                    output_ref_n25 = ref_model(us_input_ref, input_kspace_ref, mask_ref)
                loss_img_n25 = F.l1_loss(output_n25, target)
                loss_distil_n25 = multi_scale_distil_loss(output_n25, output_ref_n25) if args.distil_multi_scale else F.l1_loss(output_n25, output_ref_n25)
                loss_n25 = 0.25 * (loss_img_n25 + args.distil_alpha * loss_distil_n25)
                loss_n25.backward()
                del output_n25, output_ref_n25, us_noisy_25

                # === Pass 3: 重度噪声增广 ===
                sigma_50 = random.uniform(40, 60)
                us_noisy_50 = gaussian_noise_aug(us_input_high, sigma=sigma_50)
                output_n50 = model(us_noisy_50, input_kspace_high, mask_high)
                with torch.no_grad():
                    output_ref_n50 = ref_model(us_input_ref, input_kspace_ref, mask_ref)
                loss_img_n50 = F.l1_loss(output_n50, target)
                loss_distil_n50 = multi_scale_distil_loss(output_n50, output_ref_n50) if args.distil_multi_scale else F.l1_loss(output_n50, output_ref_n50)
                loss_n50 = 0.25 * (loss_img_n50 + args.distil_alpha * loss_distil_n50)
                loss_n50.backward()
                del output_n50, output_ref_n50, us_noisy_50

                loss = loss + loss_n25 + loss_n50
                loss_img = loss_img + loss_img_n25 + loss_img_n50
                loss_distil = loss_distil + loss_distil_n25 + loss_distil_n50

            if iter % args.report_interval == 0:
                logging.info(
                    f'  L_img={loss_img.item():.4g} L_distil={loss_distil.item():.4g} '
                    f'alpha*L_distil={args.distil_alpha * loss_distil.item():.4g}')
        else:
            # === Pass 1: 原始数据 (无蒸馏) ===
            output = model(us_input_high, input_kspace_high, mask_high)
            loss = 0.5 * F.l1_loss(output, target) if args.aug_flip else F.l1_loss(output, target)
            loss.backward()
            del output

            # === Pass 2: 轻度噪声增广 (仅当 args.aug_flip=True) ===
            if args.aug_flip:
                sigma_25 = random.uniform(15, 35)
                us_noisy_25 = gaussian_noise_aug(us_input_high, sigma=sigma_25)
                output_n25 = model(us_noisy_25, input_kspace_high, mask_high)
                loss_n25 = 0.25 * F.l1_loss(output_n25, target)
                loss_n25.backward()
                del output_n25, us_noisy_25

                # === Pass 3: 重度噪声增广 ===
                sigma_50 = random.uniform(40, 60)
                us_noisy_50 = gaussian_noise_aug(us_input_high, sigma=sigma_50)
                output_n50 = model(us_noisy_50, input_kspace_high, mask_high)
                loss_n50 = 0.25 * F.l1_loss(output_n50, target)
                loss_n50.backward()
                del output_n50, us_noisy_50

                loss = loss + loss_n25 + loss_n50

        optimizer.step()
        scheduler.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if ref_model is not None and args.distil_alpha > 0:
            avg_loss_img = 0.99 * avg_loss_img + 0.01 * loss_img.item() if iter > 0 else loss_img.item()
            avg_loss_distil = 0.99 * avg_loss_distil + 0.01 * loss_distil.item() if iter > 0 else loss_distil.item()

        if iter % args.report_interval == 0:
            if ref_model is not None and args.distil_alpha > 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                    f'Avg L_img = {avg_loss_img:.4g} Avg L_distil = {avg_loss_distil:.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
            else:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, mask_bank, acc_factors, mask_types, dataset_types):

    model.eval()
    losses = []
    psnr_list = []
    ssim_list = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            target, acc_idx, mask_idx, ds_idx = data
            target = target.unsqueeze(1).to(args.device)
            us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types)

            us_input = us_input.squeeze(1).float()  # [B, 1, 1, H, W] -> [B, 1, H, W]
            input_kspace = input_kspace.squeeze(1).float()  # [B, 1, H, W, 2] -> [B, H, W, 2]
            target = target.float()

            output = model(us_input,input_kspace,mask)

            loss = F.mse_loss(output,target)
            losses.append(loss.item())

            # 逐 slice 计算 PSNR/SSIM
            output_np = output.detach().cpu().numpy().squeeze(1)  # [B, H, W]
            target_np = target.detach().cpu().numpy().squeeze(1)
            for b in range(output_np.shape[0]):
                gt = target_np[b]
                pred = output_np[b]
                data_range = gt.max()
                if data_range > 0:
                    psnr_list.append(peak_signal_noise_ratio(gt, pred, data_range=data_range))
                    ssim_list.append(structural_similarity(gt, pred, data_range=data_range))

        avg_loss = np.mean(losses)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        writer.add_scalar('Dev_Loss', avg_loss, epoch)
        writer.add_scalar('Dev_PSNR', avg_psnr, epoch)
        writer.add_scalar('Dev_SSIM', avg_ssim, epoch)

    return avg_loss, avg_psnr, avg_ssim, time.perf_counter() - start


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
            output = model(us_input,input_kspace,mask)

            save_image(us_input, 'Input_{}'.format(datasettype_string))
            save_image(target, 'Target_{}'.format(datasettype_string))
            save_image(output, 'Reconstruction_{}'.format(datasettype_string))
            save_image(torch.abs(target.float() - output.float()), 'Error_{}'.format(datasettype_string))
            break

def _make_melora_dirname(args):
    """Build subdirectory name from MELORA_R and MELORA_TARGET."""
    r_str = args.melora_r.replace(',', '_')
    target_str = args.melora_target.replace(',', '_') if args.melora_target else 'all'
    return f'r{r_str}_{target_str}'

def save_model(args, save_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best):
    """Save LoRA adapter + training metadata to save_dir. No base weights (they don't change)."""
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora_' in k}

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

    # Adapter: LoRA weights only
    torch.save(lora_state, f=save_dir / 'adapter.pt')

    if is_new_best:
        shutil.copyfile(save_dir / 'adapter.pt', save_dir / 'adapter_best.pt')


def build_model_from_pretrained(args):
    """Load pretrained base weights, apply MELoRA, freeze base, return model with only lora trainable."""
    # Load pretrained base model
    pretrained = torch.load(args.pretrained_checkpoint)
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
    trainable_count = set_melora_trainable(model)

    # Verify and report which params are trainable
    total_count = sum(p.numel() for p in model.parameters())
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_sample = [n for n, p in model.named_parameters() if not p.requires_grad][:5]

    logger.info(f"{'='*60}")
    logger.info(f"MELoRA verification:")
    logger.info(f"  Total     params: {total_count:,}")
    logger.info(f"  Frozen    params: {frozen_count:,} ({frozen_count/total_count*100:.1f}%)")
    logger.info(f"  Trainable params: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    logger.info(f"  Trainable layers ({len(trainable_names)}):")
    for n in trainable_names[:10]:
        logger.info(f"    ✓ {n}")
    if len(trainable_names) > 10:
        logger.info(f"    ... and {len(trainable_names) - 10} more")
    logger.info(f"  Frozen layer sample:")
    for n in frozen_sample:
        logger.info(f"    ✗ {n}")
    logger.info(f"{'='*60}")

    return model


def build_ref_model(args):
    """构建冻结的参考模型（无LoRA），用于提取特征计算损失。

    注意：使用与训练模型相同的预训练权重（在低倍欠采样数据上训练得到）。
    该权重与低倍欠采样数据是配套的。
    """
    pretrained = torch.load(args.pretrained_checkpoint)
    base_state = pretrained['model']

    ref_model = DnCn(args, n_channels=1).to(args.device)
    ref_model.load_state_dict(base_state, strict=True)
    ref_model.eval()

    # 冻结全部参数，仅用于特征提取
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Reference model built from pretrained weights (frozen, no LoRA)")
    logger.info("  Note: This uses the same pretrained checkpoint as the train model")
    return ref_model


def load_model(checkpoint_file):
    """Resume from a LoRA checkpoint.pt (metadata only, no model weights)."""
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    # Load base weights from pretrained checkpoint
    pretrained = torch.load(args.pretrained_checkpoint)
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
    set_melora_trainable(model)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # Load lora adapter (same directory as checkpoint.pt)
    adapter_path = pathlib.Path(checkpoint_file).parent / 'adapter.pt'
    lora_state = torch.load(adapter_path)
    model.load_state_dict(lora_state, strict=False)

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.AdamW(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    melora_dir = args.exp_dir / 'melora' / _make_melora_dirname(args)
    melora_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(melora_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        # 保留命令行传入的新参数
        distil_alpha = args.distil_alpha
        ref_acceleration_factor = args.ref_acceleration_factor
        batch_size = args.batch_size
        args = checkpoint['args']
        args.batch_size = batch_size  # 保留传入的 batch_size
        args.distil_alpha = distil_alpha  # 保留蒸馏损失权重
        args.ref_acceleration_factor = ref_acceleration_factor  # 保留参考分支倍数
        best_psnr = checkpoint.get('best_psnr', 0.)
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model_from_pretrained(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_psnr = 0.
        start_epoch = 0

    logging.info(args)
    logging.info(model)
    train_loader, dev_loader, display1_loader = create_data_loaders(args)

    # OneCycleLR: warmup + cosine decay, 每个 batch 更新一次
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.05
    )
    # resume 时快速追赶 scheduler 到上次中断的位置
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    mask_bank = build_mask_bank(acc_factors, mask_types, dataset_types, args.usmask_path, args.device)

    # 构建参考模型用于蒸馏
    ref_model = None
    mask_bank_ref = None
    if args.distil_alpha > 0:
        ref_model = build_ref_model(args)
        ref_acc_factors = args.ref_acceleration_factor.split(',')
        mask_bank_ref = build_mask_bank(ref_acc_factors, mask_types, dataset_types,
                                        args.usmask_path, args.device)

    for epoch in range(start_epoch, args.num_epochs):

        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, scheduler, writer,
            mask_bank, acc_factors, mask_types, dataset_types,
            ref_model=ref_model, mask_bank_ref=mask_bank_ref)
        dev_loss, dev_psnr, dev_ssim, dev_time = evaluate(args, epoch, model, dev_loader, writer, mask_bank, acc_factors, mask_types, dataset_types)
        visualize(args, epoch, model, display1_loader, writer, 't1')

        is_new_best = dev_psnr > best_psnr
        best_psnr = max(best_psnr, dev_psnr)
        save_model(args, melora_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} PSNR = {dev_psnr:.4g} SSIM = {dev_ssim:.4g} '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
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

    # Distillation settings
    parser.add_argument('--distil-alpha', type=float, default=0.1,
                        help='Weight for distillation loss (L1 between main and ref output, 0 to disable)')
    parser.add_argument('--ref-acceleration-factor', type=str, default='4x',
                        help='Acceleration factor for the reference branch (低倍, e.g., 4x)')
    parser.add_argument('--aug-flip', action='store_true', default=False,
                        help='If set, augment data with Gaussian noise (sigma=25, sigma=50): triples effective batch with three forward/backward passes')
    parser.add_argument('--distil-multi-scale', action='store_true', default=False,
                        help='If set, compute distillation loss at 3 scales (0.5x, 1x, 2x) and average')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
