import argparse
import pathlib
import re
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return structural_similarity(gt,pred,multichannel=True, data_range=gt.max())

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn
)
METRIC_NAMES = sorted(METRIC_FUNCS)


def format_metric_summary(means, stddevs=None):
    if stddevs is None:
        return ' '.join(
            f'{name} = {means[name]:.4g}' for name in METRIC_NAMES
        )

    return ' '.join(
        f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}'
        for name in METRIC_NAMES
    )


def parse_metric_means(report_line):
    means = {}
    number_pattern = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'

    for name in METRIC_NAMES:
        match = re.search(rf'{name}\s*=\s*{number_pattern}', report_line)
        if match is None:
            return None
        means[name] = float(match.group(1))

    return means


def write_aggregated_report(report_file, acc_factor, metrics_report):
    current_line = f'[{acc_factor}]   {metrics_report}'
    report_lines = []

    if report_file.exists():
        report_lines = [
            line.strip() for line in report_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    updated_lines = []
    replaced_current = False

    for line in report_lines:
        if line.startswith('[AVG]'):
            continue
        if line.startswith(f'[{acc_factor}]'):
            if not replaced_current:
                updated_lines.append(current_line)
                replaced_current = True
            continue
        updated_lines.append(line)

    if not replaced_current:
        updated_lines.append(current_line)

    metric_rows = []
    for line in updated_lines:
        parsed_means = parse_metric_means(line)
        if parsed_means is not None:
            metric_rows.append(parsed_means)

    if metric_rows:
        overall_means = {
            name: float(np.mean([row[name] for row in metric_rows]))
            for name in METRIC_NAMES
        }
        updated_lines.append(f'[AVG]   {format_metric_summary(overall_means)}')

    report_file.write_text('\n'.join(updated_lines), encoding='utf-8')


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        return format_metric_summary(means, stddevs)




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        if tgt_file.suffix != '.h5':
            continue
        #print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            target = target[recons_key]
            target = np.array(target)
            recons = recons['reconstruction']
            recons = np.transpose(recons,[1,2,0])
            print(tgt_file)
            print (target.shape,recons.shape)
            print (type(target),type(recons))
            metrics.push(target, recons)
            
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--acc-factor', type=str, required=True) 
    parser.add_argument('--report-file-acc-factor', type=str, default=None,
                        help='Acc factor used in the report filename when aggregating multiple runs')
    parser.add_argument('--mask-type', type=str, required=True)
    parser.add_argument('--dataset-type', type=str, required=True)
    args = parser.parse_args()

    recons_key = 'volfs'
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()

    report_acc_factor = args.report_file_acc_factor or args.acc_factor
    report_file = args.report_path / 'report_{}_{}_{}.txt'.format(
        args.dataset_type, args.mask_type, report_acc_factor
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if args.report_file_acc_factor is None:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(metrics_report)
    else:
        write_aggregated_report(report_file, args.acc_factor, metrics_report)

    #print(metrics)
