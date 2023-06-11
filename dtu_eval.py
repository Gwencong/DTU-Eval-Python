import argparse
from utils import compute_scans

scans = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
parser = argparse.ArgumentParser()
parser.add_argument('--scans', type=list, default=scans, help="scans to be evalutation")
parser.add_argument('--method', type=str, default='mvsnet', help="method name, such as mvsnet,casmvsnet")
parser.add_argument('--pred_dir', type=str, default='./Predict/mvsnet', help="predict result ply file path")
parser.add_argument('--gt_dir', type=str, default='./SampleSet/MVS Data',help="groud truth ply file path")
parser.add_argument('--down_dense', type=float, default=0.2, help="downsample density, Min dist between points when reducing")
parser.add_argument('--patch', type=float, default=60, help="patch size")
parser.add_argument('--max_dist', type=float, default=20, help="outlier thresshold of 20 mm")
parser.add_argument('--vis', type=bool, default=False, help="visualization")
parser.add_argument('--vis_thresh', type=float, default=10, help="visualization distance threshold of 10mm")
parser.add_argument('--vis_out_dir', type=str, default="./visualize_outs", help="visualization result save dir")
args = parser.parse_args()

if __name__ == "__main__":
    # args.scans    = [1]
    # args.pred_dir = "/home/gwc/gwc/data/DTU/eval/Predict/r-mvsnet"
    # args.gt_dir   = "/home/gwc/gwc/data/DTU/eval/SampleSet/MVS Data"

    scans    = args.scans
    method   = args.method
    pred_dir = args.pred_dir
    gt_dir   = args.gt_dir
    vis      = args.vis

    exclude = ["scans", "method", "pred_dir", "gt_dir"]
    args = vars(args)
    args = {key:args[key] for key in args if key not in exclude}
    acc, comp, overall = compute_scans(scans, method, pred_dir, gt_dir, **args)
    print(f"mean acc:{acc:>12.4f}\nmean comp:{comp:>11.4f}\nmean overall:{overall:>8.4f}")
