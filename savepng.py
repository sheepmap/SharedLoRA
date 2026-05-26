# savepng.py
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def ensure_axial_format(data):
    """
    将3D数据统一转为 (D, H, W) 格式，确保 data[i] 输出的是 Axial（俯视）切片。
    原理：医学图像中，层数(D)通常是空间分辨率最小的维度。
    """
    if len(data.shape) != 3:
        return data
    
    # 找到最小维度的索引，作为 D (层数)
    d_axis = np.argmin(data.shape)
    
    if d_axis != 0:
        # 将 d_axis 移动到第 0 维
        data = np.moveaxis(data, d_axis, 0)
        
    return data

def h5_to_png(h5_dir, output_dir, prefix='img', slices_per_row=8, mode='recon'):
    """
    将 H5 文件批量转换为 PNG 图片
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
    
    for h5_file in tqdm(h5_files, desc='Converting'):
        h5_path = os.path.join(h5_dir, h5_file)
        
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())
            
            if mode == 'recon' or (mode == 'all' and 'reconstruction' in keys):
                data = f['reconstruction'][:]
                prefix_name = 'recon'
            elif mode == 'raw' or (mode == 'all' and 'volfs' in keys):
                data = f['volfs'][:]
                prefix_name = 'gt'
            elif mode == 'all' and 'img_volus_4x' in keys:
                data = f['img_volus_4x'][:]
                prefix_name = 'us'
            elif mode == 'all' and 'img_volus_8x' in keys:
                data = f['img_volus_8x'][:]
                prefix_name = 'us'
            else:
                print(f'  Skip {h5_file}: unknown keys {keys}')
                continue
        
        # 统一处理为 (D, H, W)，确保无论原本是什么排列，输出的都是俯视图
        data = ensure_axial_format(data)
        
        base_name = os.path.splitext(h5_file)[0]
        
        # 每个切片单独保存 PNG
        slice_dir = os.path.join(output_dir, base_name)
        os.makedirs(slice_dir, exist_ok=True)
        
        for i in range(data.shape[0]):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data[i], cmap='gray')
            ax.set_title(f'{base_name} - Slice {i}', fontsize=10)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(slice_dir, f'slice_{i:03d}.png'), dpi=100)
            plt.close(fig)
        
        # 蒙太奇形式
        n_slices = data.shape[0]
        n_rows = (n_slices + slices_per_row - 1) // slices_per_row
        
        fig, axes = plt.subplots(n_rows, slices_per_row, figsize=(slices_per_row * 2, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_slices):
            row = i // slices_per_row
            col = i % slices_per_row
            axes[row, col].imshow(data[i], cmap='gray')
            axes[row, col].set_title(f'{i}', fontsize=6)
            axes[row, col].axis('off')
        
        for i in range(n_slices, n_rows * slices_per_row):
            row = i // slices_per_row
            col = i % slices_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_{prefix_name}_{base_name}_montage.png'), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    # 自动检测
    h5_dir = '/root/autodl-tmp/SHFormer-master/datasets/ixi_t2/cartesian/test/acc_4x'
    output_dir = '/root/autodl-tmp/SHFormer-master/datasets/ixi_t2/cartesian/test/png'
    mode = 'all'
    
    h5_to_png(h5_dir, output_dir, mode=mode)
    
    print(f'\n✅ Done! PNGs saved to: {output_dir}')
