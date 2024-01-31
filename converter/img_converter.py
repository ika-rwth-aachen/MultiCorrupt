import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
import multiprocessing as mp
import pickle
import copy
from pathlib import Path
import img

IMG_CORRUPTIONS = ['snow', 'fog', 'temporalmisalignment', 'brightness', 'dark',
                   'missingcamera', 'motionblur']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate corrupted nuScenes dataset for Images')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int,
                        default=4)
    parser.add_argument('-a', '--corruption', help='corruption type', type=str,
                        choices=IMG_CORRUPTIONS, default='fog')
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='/workspace/nuscenes/nuscenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='/workspace/multicorrupt/fog')
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

    imageset = os.path.join(args.root_folder, "deepinter_infos_val.pkl")

    with open(imageset, 'rb') as f:
        infos = pickle.load(f)

    all_files = infos['infos']
    all_paths = copy.deepcopy(all_files)

    Path(args.dst_folder).mkdir(parents=True, exist_ok=True)
    lidar_save_root = os.path.join(args.dst_folder, 'samples/LIDAR_TOP')
    if not os.path.exists(lidar_save_root):
        os.makedirs(lidar_save_root)

    def _map(i: int) -> None:
        info = all_paths[i]
        data_paths = [cam_info['data_path'] for cam_info in info['cams'].values()]

        if args.corruption == 'temporalmisalignment':
            s = [0.2, 0.4, 0.6][args.severity - 1]
            for j in range(6):
                if np.random.rand() < s and i>=1:
                    prev_info = all_paths[i-1]
                    prev_data_paths = [cam_info['data_path'] for cam_info in prev_info['cams'].values()]
                    new_data_path = prev_data_paths[j][16:] 
                else:
                    new_data_path = data_paths[j][16:]

                full_input_path = os.path.join(args.root_folder, new_data_path)
                full_output_path = os.path.join(args.dst_folder, data_paths[j][16:])
                
                os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
                cv2.imwrite(full_output_path, cv2.imread(full_input_path, cv2.IMREAD_UNCHANGED))
        else:
            for path in data_paths:
                new_data_path = path[16:]
                image = cv2.imread((os.path.join(args.root_folder, new_data_path)), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if args.corruption == 'snow':
                    new_image = img.snow(image, args.severity)

                elif args.corruption == 'fog':
                    new_image = img.fog(image, args.severity)

                elif args.corruption == 'brightness':
                    new_image = img.brightness(image, args.severity)

                elif args.corruption == 'motionblur':
                    new_image = img.img_motion_blur(image, args.severity)

                elif args.corruption == 'dark':
                    new_image = img.low_light(image, args.severity)

                elif args.corruption == 'missingcamera':
                    s = [0.2, 0.4, 0.6][args.severity - 1]
                    if np.random.rand() < s:
                        new_image = np.zeros_like(image)
                        new_image = new_image
                    else:
                        new_image = image
                else:
                    raise NotImplementedError('Corruption not implemented')

                new_image = new_image.astype(np.uint8)
                os.makedirs(os.path.dirname(os.path.join(args.dst_folder, new_data_path)), exist_ok=True)
                cv2.imwrite(os.path.join(args.dst_folder, new_data_path), cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

    n = len(all_files)
    with mp.Pool(processes=args.n_cpus) as pool:
        l = list(tqdm(pool.imap(_map, range(n)), total=n))
