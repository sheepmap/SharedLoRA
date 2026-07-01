import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch,CreateZeroFilledImageFn

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid,mask_path,
                 data_acceleration_factor=None): # acc_factor can be passed here and saved as self variable
        #files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.mask_path = mask_path
        self.acc_factors = acc_factors
        self.mask_types = mask_types
        self.dataset_types = dataset_types
        self.data_acceleration_factor = data_acceleration_factor
        for dataset_type in dataset_types:
            dataroot = os.path.join(root, dataset_type)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot, mask_type,train_or_valid)
                for acc_factor in acc_factors:
                    read_acc_factor = data_acceleration_factor or acc_factor
                    acc_root = self._resolve_acc_root(newroot, read_acc_factor)
                    files = [f for f in pathlib.Path(acc_root).iterdir() if f.suffix == '.h5']
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]

    @staticmethod
    def _resolve_acc_root(base_root, acc_factor):
        requested_root = pathlib.Path(os.path.join(base_root, 'acc_{}'.format(acc_factor)))
        if requested_root.exists():
            return str(requested_root)

        fallback_16x = pathlib.Path(os.path.join(base_root, 'acc_16x'))
        if fallback_16x.exists():
            return str(fallback_16x)

        acc_roots = sorted(
            p for p in pathlib.Path(base_root).iterdir()
            if p.is_dir() and p.name.startswith('acc_')
        )
        if acc_roots:
            return str(acc_roots[0])

        raise FileNotFoundError(
            f'No acc_* directory found under {base_root}. '
            f'Requested acc_{acc_factor}, and acc_16x fallback was also missing.'
        )



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__

        fname, slice, acc_factor, mask_type, dataset_type = self.examples[i]

        with h5py.File(fname, 'r') as data:
            target = data['volfs'][:,:,slice].astype(np.float64)

        acc_idx = self.acc_factors.index(acc_factor)
        mask_idx = self.mask_types.index(mask_type)
        ds_idx = self.dataset_types.index(dataset_type)
        return torch.from_numpy(target), acc_idx, mask_idx, ds_idx

            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type,mask_type,mask_path):

        # List the h5 files in root
        files = [f for f in pathlib.Path(root).iterdir() if f.suffix == '.h5']
        self.examples = []
        self.mask_path = mask_path

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type, dataset_type = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            mask = np.load(os.path.join(self.mask_path,dataset_type, mask_type,'mask_{}.npy'.format(acc_factor)))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(mask), str(fname.name),slice

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,dataset_type,mask_type,acc_factor,mask_path):

        newroot = os.path.join(root, dataset_type,mask_type,'validation','acc_{}'.format(acc_factor))
        # List the h5 files in root 
        files = [f for f in pathlib.Path(newroot).iterdir() if f.suffix == '.h5']
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type

        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        mask_path = os.path.join(mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor))
        self.mask_path = mask_path

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print(hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)
            mask = np.load(self.mask_path)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),mask
 
