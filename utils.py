import os
import time
import numpy as np

from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
from sklearn import neighbors as skln
from plyfile import PlyData, PlyElement


def read_ply(file):
    data = PlyData.read(file)
    vertex = data['vertex']
    data_pcd = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return data_pcd

def write_vis_pcd(file, points, colors):
    points = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors = np.array([tuple(v) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(points), points.dtype.descr + colors.dtype.descr)
    for prop in points.dtype.names:
        vertex_all[prop] = points[prop]
    for prop in colors.dtype.names:
        vertex_all[prop] = colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(file)


def comput_one_scan(scanid,             # the scan id to be computed 
                    pred_ply,           # predict points cloud file path, such as "./mvsnet001_l3.ply"
                    gt_ply,             # ground truth points cloud file path, such as "./stl001_total.ply"
                    mask_file,          # obsmask file path, decide which parts of 3D space should be used for evaluation
                    plane_file,         # plane file path, used to destinguise which Stl points are 'used'
                    down_dense  = 0.2,  # downsample density, Min dist between points when reducing
                    patch       = 60,   # patch size
                    max_dist    = 20,   # outlier thresshold of 20 mm
                    vis         = False,# whether save distance visualization result 
                    vis_thresh  = 10,   # visualization distance threshold of 10mm
                    vis_out_dir = "./visualize_outs"):
    '''Compute accuracy(mm), completeness(mm), overall(mm) for one scan 

        scanid:         the scan id to be computed 
        pred_ply:       predict points cloud file path, such as "./mvsnet001_l3.ply"
        gt_ply:         ground truth points cloud file path, such as "./stl001_total.ply"
        mask_file:      obsmask file path, decide which parts of 3D space should be used for evaluation
        plane_file:     plane file path, used to destinguise which Stl points are 'used'
        down_dense:     downsample density, Min dist between points when reducing
        patch:          patch size
        max_dist:       outlier thresshold of 20 mm
        vis:            whether save distance visualization result 
        vis_thresh:     visualization distance threshold of 10mm
        vis_out_dir:    visualization result save directory
    '''
    
    thresh = down_dense
    pbar = tqdm(total=8)
    pbar.set_description(f'[scan{scanid}] read data pcd')
    data_pcd = read_ply(pred_ply)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] masking data pcd')
    obs_mask_file = loadmat(mask_file)
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] read STL pcd')
    stl = read_ply(gt_ply)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute stl2data')
    ground_plane = loadmat(plane_file)['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] visualize error')
    if vis:
        Path(vis_out_dir).mkdir(parents=True, exist_ok=True)
        vis_dist = vis_thresh
        R = np.array([[255,0,0]], dtype=np.float64)
        G = np.array([[0,255,0]], dtype=np.float64)
        B = np.array([[0,0,255]], dtype=np.float64)
        W = np.array([[255,255,255]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{vis_out_dir}/vis_{scanid:03}_d2s.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{vis_out_dir}/vis_{scanid:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(f"\t\t\tacc.(mm):{mean_d2s:.4f}, comp.(mm):{mean_s2d:.5f}, overall(mm):{over_all:.4f}")
    return mean_d2s, mean_s2d, over_all

def compute_scans(scans, method, pred_dir, gt_dir, **kargs):
    t1 = time.time()
    acc ,comp ,overall = [], [], []
    for scanid in scans:
        pred_ply    = os.path.join(pred_dir, f"{method}{scanid:03}_l3.ply")   
        gt_ply      = os.path.join(gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
        mask_file   = os.path.join(gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
        plane_file  = os.path.join(gt_dir, f'ObsMask/Plane{scanid}.mat')
        assert os.path.exists(pred_ply),   f"File '{pred_ply}' not found"
        assert os.path.exists(gt_ply),     f"File '{gt_ply}' not found"
        assert os.path.exists(mask_file),  f"File '{mask_file}' not found"
        assert os.path.exists(plane_file), f"File '{plane_file}' not found"
        result = comput_one_scan(scanid, pred_ply, gt_ply, mask_file, plane_file, **kargs)
        acc.append(result[0])
        comp.append(result[1])
        overall.append(result[2])
    mean_acc = np.mean(acc)
    mean_comp = np.mean(comp)
    mean_overall = np.mean(overall)
    t2 = time.time()
    print(f"Finished, total time cost: {t2-t1:.2f}s")
    return mean_acc, mean_comp, mean_overall

