import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import DnCn
import h5py
from tqdm import tqdm

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
    out_dir.mkdir(exist_ok=True)
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


def load_model(checkpoint_file, use_lora=False, lora_path=None):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = DnCn(args,n_channels=1).to(args.device)

    if use_lora and lora_path is not None:
        # LoRA inference: load base weights, apply MELoRA, load adapter, merge
        model.load_state_dict(checkpoint['model'], strict=False)

        # Load LoRA hyperparams from checkpoint.pt (saved alongside adapter)
        lora_chk = torch.load(pathlib.Path(lora_path).parent / 'checkpoint.pt')
        lora_args = lora_chk['args']
        melora_r = [int(x.strip()) for x in lora_args.melora_r.split(",")]
        melora_alpha = [int(x.strip()) for x in lora_args.melora_alpha.split(",")]
        target = [x.strip() for x in lora_args.melora_target.split(",")] if getattr(lora_args, 'melora_target', None) else None
        from MC_DDPM_SH.models.melora_utils import apply_melora_to_model
        apply_melora_to_model(model, melora_r, melora_alpha,
                              lora_dropout=getattr(lora_args, 'melora_dropout', 0.0),
                              target_module_names=target,
                              verbose=True)

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        lora_state = torch.load(lora_path)
        model.load_state_dict(lora_state, strict=False)

        for m in model.modules():
            if hasattr(m, 'merge'):
                m.merge()
                m.disable_adapters = True
        print(f"MELoRA adapter loaded from {lora_path} and merged for inference")
    else:
        # Original inference (no LoRA)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'], strict=False)

    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            us_input, input_kspace, target,mask,fnames,slices = data
            us_input = us_input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.to(args.device)
            mask = mask.to(args.device)

            us_input = us_input.float()

            recons = model(us_input,input_kspace,mask).to('cpu').squeeze(1)

            if args.dataset_type == 'cardiac':
                recons = recons[:,5:155,5:155]

            
            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint, args.use_lora, args.lora_path)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


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
