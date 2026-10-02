import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,SliceDisplayDataDev
from models import DnCn
from utils import cartesian_mask, gaussian_mask
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
    # Reuse the validation dataset for visualization: SliceDisplayDataDev reads
    # precomputed img_volus_<af>/kspace_volus_<af> keys, which --volfs-only h5
    # files do not contain. visualize() takes the mask_bank branch and
    # synthesizes the undersampled inputs on the GPU instead.
    display1_data = dev_data

    return dev_data, train_data, display1_data

def create_data_loaders(args):
    dev_data, train_data, display1_data = create_datasets(args)

    display1 = [display1_data[i] for i in range(0, len(display1_data), len(display1_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    display_loader1 = DataLoader(
        dataset=display1,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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


def gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=False):
    """Perform FFT, masking, and IFFT on GPU.

    mask_resample restores the original SHFormer training protocol: a fresh
    random mask is drawn per sample with the original generators
    (cartesian_mask/gaussian_mask) instead of reusing one fixed pattern, so
    the network learns the sampling distribution rather than a single
    realization. Masks are generated in memory only -- nothing is saved.
    """
    B = target.shape[0]
    masks = []
    for i in range(B):
        di = ds_idx[i].item() if hasattr(ds_idx[i], 'item') else ds_idx[i]
        mi = mask_idx[i].item() if hasattr(mask_idx[i], 'item') else mask_idx[i]
        ai = acc_idx[i].item() if hasattr(acc_idx[i], 'item') else acc_idx[i]
        if mask_resample:
            h, w = target.shape[-2], target.shape[-1]
            acc_val = float(str(acc_factors[ai]).rstrip('x').replace('_', '.'))
            if mask_types[mi] == 'cartesian':
                m = cartesian_mask((h, w), acc_val)
            else:
                m = gaussian_mask((h, w), acc_val)
            masks.append(torch.from_numpy(np.asarray(m)).to(target.device))
        else:
            key = (dataset_types[di], mask_types[mi], acc_factors[ai])
            masks.append(mask_bank[key])
    mask = torch.stack(masks)                                          # (B, H, W)
    kspace = torch.fft.fft2(target, norm='ortho')                     # GPU FFT
    us_kspace = kspace * mask.unsqueeze(1)
    us_img = torch.abs(torch.fft.ifft2(us_kspace, norm='ortho'))      # GPU IFFT
    return us_img.unsqueeze(1), torch.view_as_real(us_kspace), mask


def train_epoch(args, epoch, model,data_loader, optimizer, writer, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=False):

    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        target, acc_idx, mask_idx, ds_idx = data
        target = target.unsqueeze(1).to(args.device)
        us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=mask_resample)

        us_input = us_input.squeeze(1).float()  # [B, 1, 1, H, W] -> [B, 1, H, W]
        input_kspace = input_kspace.squeeze(1).float()  # [B, 1, H, W, 2] -> [B, H, W, 2]
        target = target.float()
        output = model(us_input,input_kspace,mask)

        loss = F.l1_loss(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=False):

    model.eval()
    losses = []
    psnr_list = []
    ssim_list = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            target, acc_idx, mask_idx, ds_idx = data
            target = target.unsqueeze(1).to(args.device)
            us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=mask_resample)

            us_input = us_input.squeeze(1).float()  # [B, 1, 1, H, W] -> [B, 1, H, W]
            input_kspace = input_kspace.squeeze(1).float()  # [B, 1, H, W, 2] -> [B, H, W, 2]
            target = target.float()

            output = model(us_input,input_kspace,mask)

            loss = F.mse_loss(output,target)
            losses.append(loss.item())

            # Compute PSNR/SSIM per slice
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


def visualize(args, epoch, model, data_loader, writer, datasettype_string, mask_bank=None, acc_factors=None, mask_types=None, dataset_types=None, mask_resample=False):


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
                us_input, input_kspace, mask = gpu_undersample(target, acc_idx, mask_idx, ds_idx, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=mask_resample)
                us_input = us_input.squeeze(1).float()  # [B, 1, 1, H, W] -> [B, 1, H, W]
                input_kspace = input_kspace.squeeze(1).float()  # [B, 1, H, W, 2] -> [B, H, W, 2]
            else:
                # SliceDisplayDataDev returns (input_img, input_kspace, target, mask)
                us_input, input_kspace, target, mask = data
                us_input = us_input.unsqueeze(1).to(args.device).float()  # [B, H, W] -> [B, 1, H, W]
                input_kspace = input_kspace.to(args.device).float()
                target = target.unsqueeze(1).to(args.device)
                mask = mask.to(args.device)

            target = target.float()
            output = model(us_input,input_kspace,mask)

            save_image(us_input, 'Input_{}'.format(datasettype_string))
            save_image(target, 'Target_{}'.format(datasettype_string))
            save_image(output, 'Reconstruction_{}'.format(datasettype_string))
            save_image(torch.abs(target.float() - output.float()), 'Error_{}'.format(datasettype_string))
            break

def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):

    model = DnCn(args,n_channels=1).to(args.device)
    return model

def load_model(checkpoint_file):
    # PyTorch >= 2.6 defaults weights_only=True, which rejects checkpoints
    # containing argparse.Namespace; our own checkpoints are trusted.
    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
    except TypeError:  # older PyTorch without the kwarg
        checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'], strict=False)

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        ckpt_args = checkpoint['args']
        best_psnr = checkpoint.get('best_psnr', 0.)
        start_epoch = checkpoint['epoch'] + 1

        lr_changed = abs(args.lr - ckpt_args.lr) > 1e-12
        remaining = max(1, args.num_epochs - start_epoch)
        if lr_changed:
            # New LR from the command line: jump to it immediately and re-run
            # cosine annealing over the remaining epochs.
            print(f'LR changed on resume: {ckpt_args.lr} -> {args.lr}; '
                  f'new cosine over remaining {remaining} epochs')
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=args.lr_eta_min)
        elif 'scheduler' in checkpoint:
            # Same LR: restore the saved trajectory exactly as before.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ckpt_args.num_epochs, eta_min=args.lr_eta_min)
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # Old checkpoints carry no scheduler state: fast-forward it and
            # override the LR with the command-line value.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr_eta_min)
            for _ in range(start_epoch):
                scheduler.step()
            optimizer.param_groups[0]['lr'] = args.lr
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        # CosineAnnealingLR instead of StepLR: with lr_step_size=40 the LR would not
        # decay even once in short runs (e.g. 10 epochs); cosine decays smoothly.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr_eta_min)
        best_psnr = 0.
        start_epoch = 0

    logging.info(args)
    logging.info(model)
    train_loader, dev_loader, display1_loader = create_data_loaders(args)

    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    # With --mask-resample the masks are generated fresh per sample inside
    # gpu_undersample; no pre-loaded bank is needed (or required to exist).
    mask_bank = {} if args.mask_resample else \
        build_mask_bank(acc_factors, mask_types, dataset_types, args.usmask_path, args.device)

    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=args.mask_resample)
        dev_loss, dev_psnr, dev_ssim, dev_time = evaluate(args, epoch, model, dev_loader, writer, mask_bank, acc_factors, mask_types, dataset_types, mask_resample=args.mask_resample)
        visualize(args, epoch, model, display1_loader, writer, 't1',
                  mask_bank, acc_factors, mask_types, dataset_types, mask_resample=args.mask_resample)
        scheduler.step()

        is_new_best = dev_psnr > best_psnr
        best_psnr = max(best_psnr, dev_psnr)
        save_model(args, args.exp_dir, epoch, model, optimizer, scheduler, best_psnr, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} PSNR = {dev_psnr:.4g} SSIM = {dev_ssim:.4g} '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr-eta-min', type=float, default=1e-7,
                        help='Minimum learning rate for cosine annealing')
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
    parser.add_argument('--mask-resample', action='store_true',
                        help='Draw a fresh random mask per training/validation sample with the '
                             'original generators (nothing is saved) so the model learns the '
                             'sampling distribution instead of one fixed pattern; mask_bank is '
                             'then not required')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
