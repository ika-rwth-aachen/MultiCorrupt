import os
import copy
import pickle
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from nuscenes import NuScenes
import lidar
import warnings

warnings.filterwarnings('ignore')
LIDAR_CORRUPTIONS = ['pointsreducing', 'beamsreducing', 'snow', 'fog', 'copy',
                     'spatialmisalignment', 'temporalmisalignment', 'motionblur']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate corrupted nuScenes dataset for LiDAR')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int,
                        default=4)
    parser.add_argument('-a', '--corruption', help='corruption type', type=str,
                        choices=LIDAR_CORRUPTIONS, default='beamsreducing')
    parser.add_argument('-s', '--sweep', help='if apply for sweep LiDAR', type=bool,
                        default=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='/workspace/nuscenes/nuscenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='/workspace/multicorrupt/beamsreducing/1')  
    parser.add_argument('-f', '--severity', help='severity level {1,2,3}', type=int,
                        default=1)
    parser.add_argument('--seed', help='random seed', type=int,
                        default=1000)
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = parse_arguments()

    print(f'using {args.n_cpus} CPUs')
    print(f'using {args.seed} as numpy random seed')
    np.random.seed(args.seed)

    nusc_info = NuScenes(version='v1.0-trainval', dataroot=args.root_folder, verbose=True)

    imageset = os.path.join(args.root_folder, 'deepinter_infos_val.pkl')
    with open(imageset, 'rb') as f:
            infos = pickle.load(f)
    all_files = infos['infos']
    all_paths =  copy.deepcopy(all_files)
    
    Path(args.dst_folder).mkdir(parents=True, exist_ok=True)
    lidar_save_root = os.path.join(args.dst_folder , 'samples/LIDAR_TOP')
    if not os.path.exists(lidar_save_root):
        os.makedirs(lidar_save_root)
    
    if args.sweep:
        sweep_root = os.path.join(args.dst_folder , 'sweeps/LIDAR_TOP')
        if not os.path.exists(sweep_root):
            os.makedirs(sweep_root)


    def sweep_map(i: int) -> None:
        info = all_paths[i]
        lidar_path = info['lidar_path'][16:]
        point = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
        sweep_info = info['sweeps']
        
        if args.corruption == 'pointsreducing':
            new_point = lidar.simulate_missing_lidar_points(point, args.severity)
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_sweep_point = lidar.simulate_missing_lidar_points(sweep_point, args.severity)
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'beamsreducing':
            new_point = lidar.reduce_LiDAR_beamsV2(point, args.severity)
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_sweep_point = lidar.reduce_LiDAR_beamsV2(sweep_point, args.severity)
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'motionblur':
            new_point = lidar.pts_motion(point, args.severity)
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_sweep_point = lidar.pts_motion(sweep_point, args.severity)
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'snow':
            lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
            lidar_seg_label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
            lidarseg_labels_filename = os.path.join(args.root_folder, lidar_seg_label_path)
            label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        
            _, new_point, _ = lidar.simulate_snow(
                point,
                label,
                args.severity,
                noise_floor=0.7,
                beam_divergence=float(np.degrees(3e-3)),
                shuffle=True
            )

            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                _, new_sweep_point = lidar.simulate_snow_sweep(
                    sweep_point,
                    args.severity,
                    noise_floor=0.7,
                    beam_divergence=float(np.degrees(3e-3)),
                    shuffle=True
                )
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'fog':
            new_point, _, _,  _ = lidar.simulate_fog(args.severity, point, 10)
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_sweep_point, _, _,  _ = lidar.simulate_fog(args.severity, sweep_point,  10)
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'spatialmisalignment':
            new_point = lidar.transform_points(point, args.severity)
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_sweep_point = lidar.transform_points(sweep_point, args.severity)
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                new_sweep_point.astype(np.float32).tofile(sweep_path)

        elif args.corruption == 'temporalmisalignment':
            sweep_list = []
            if len(sweep_info) == 0:
                pass
            else:
                for m in sweep_info:
                    sweep_m_path = m['data_path'][16:]
                    sweep_list.append(sweep_m_path)

            if i >= 2:
                prev_info = all_paths[i-1]
                prev_sweep_list = []
                prev_sweep_info=prev_info['sweeps']
                if len(prev_sweep_info) == 0:
                    pass
                else:
                    for m in prev_sweep_info:
                        prev_sweep_m_path = m['data_path'][16:]
                        prev_sweep_list.append(prev_sweep_m_path)

            s = [0.2, 0.4, 0.6][args.severity - 1]

            if np.random.rand() < s and i >= 2:
                prev_lidar_path = prev_info['lidar_path'][16:]
                prev_point = np.fromfile(os.path.join(args.root_folder, prev_lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_point = prev_point.copy()

                if len(prev_sweep_list) != len(sweep_list):
                    for n in sweep_list:
                        new_sweep_point = np.fromfile(os.path.join(args.root_folder, n), dtype=np.float32, count=-1).reshape([-1, 5]) 
                        sweep_path = os.path.join(args.dst_folder, n)
                        new_sweep_point.astype(np.float32).tofile(sweep_path)
                else:  
                    prev_points = []
                    for n in prev_sweep_list:
                        new_sweep_point = np.fromfile(os.path.join(args.root_folder, n), dtype=np.float32, count=-1).reshape([-1, 5]) 
                        prev_points.append(new_sweep_point)

                    for idx, n in enumerate(sweep_list):
                        sweep_path = os.path.join(args.dst_folder, n)
                        prev_points[idx].astype(np.float32).tofile(sweep_path)

            else:
                new_point = point.copy()
                for n in sweep_list:
                    new_sweep_point = np.fromfile(os.path.join(args.root_folder, n), dtype=np.float32, count=-1).reshape([-1, 5]) 
                    sweep_path = os.path.join(args.dst_folder, n)
                    new_sweep_point.astype(np.float32).tofile(sweep_path)
        
        elif args.corruption == "copy":
            new_point = point
            for n in sweep_info:
                sweep_path = n['data_path'][16:]
                sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                sweep_path = os.path.join(args.dst_folder, sweep_path)
                sweep_point.astype(np.float32).tofile(sweep_path)
        
        else: 
            raise NotImplementedError('Corruption not supported')

        lidar_save_path = os.path.join(args.dst_folder, lidar_path)
        new_point.astype(np.float32).tofile(lidar_save_path)


    def sample_map(i: int) -> None:
        info = all_paths[i]
        lidar_path = info['lidar_path'][16:]
        point = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
        
        if args.corruption == 'pointsreducing':
            new_point = lidar.simulate_missing_lidar_points(point, args.severity)

        elif args.corruption == 'beamsreducing':
            new_point = lidar.reduce_LiDAR_beamsV2(point, args.severity)

        elif args.corruption == 'motionblur':
            new_point = lidar.pts_motion(point, args.severity)

        elif args.corruption == 'snow':
            lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
            label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
            lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
            label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            
            _, new_point, _ = lidar.simulate_snow(
                point,
                label, 
                args.severity,
                noise_floor=0.7,
                beam_divergence=float(np.degrees(0.003)),
                shuffle=True
            )

        elif args.corruption == 'fog':
            new_point, _, _,  _ = lidar.simulate_fog(args.severity, point, 10)

        elif args.corruption == 'spatialmisalignment':
            new_point = lidar.transform_points(point, args.severity)
        
        elif args.corruption == 'temporalmisalignment':
            if i>=2:
                prev_info = all_paths[i-1]

            s = [0.2, 0.4, 0.6][args.severity - 1]

            if np.random.rand() < s and i >= 2:
                prev_lidar_path = prev_info['lidar_path'][16:]
                prev_point = np.fromfile(os.path.join(args.root_folder, prev_lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 
                new_point = prev_point.copy()

            else:
                new_point = point.copy()
                
        elif args.corruption == "copy":
            new_point = point
        
        else:
            raise NotImplementedError('Corruption not supported')

        lidar_save_path = os.path.join(args.dst_folder, lidar_path)
        new_point.astype(np.float32).tofile(lidar_save_path)


    length = len(all_files)
    if args.sweep:
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(sweep_map, range(length)), total=length))

    else:
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(sample_map, range(length)), total=length))
