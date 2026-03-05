import os
import shutil
from nuscenes.nuscenes import NuScenes
import argparse

def parse_arguments():
    valid_plot_types = ['front_camera', 'multi_view_camera']
    parser = argparse.ArgumentParser(description='Export nuScenes scene images to a target destination')
    parser.add_argument('-c', '--corruption_type', help='Corruption type', type=str, default='motionblur')
    parser.add_argument('-s', '--severity_level', help='Severity level (1, 2, 3)', type=str, default='1')
    parser.add_argument('-p', '--plot_type', help='Plot type', type=str, default='multi_view_camera', choices=valid_plot_types)
    parser.add_argument('-n', '--scene_name', help='Scene name', type=str, default='scene-0100')
    parser.add_argument('-o', '--output_dir', help='Output directory', type=str, default='exported_scenes')
    arguments = parser.parse_args()
    return arguments

def get_scene_index_by_name(nusc, scene_name):
    for i, scene in enumerate(nusc.scene):
        if scene['name'] == scene_name:
            return i
    return None

def copy_image_files_for_sample(nusc, sample_token, output_dir, plot_type):
    if plot_type == "front_camera":
        camera_channels = ['CAM_FRONT']
    elif plot_type == "multi_view_camera":
        camera_channels = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    
    # Copy camera images
    for channel in camera_channels:
        camera_token = nusc.get('sample', sample_token)['data'][channel]
        camera_data = nusc.get('sample_data', camera_token)
        source_path = os.path.join(nusc.dataroot, camera_data['filename'])
        target_path = os.path.join(output_dir, camera_data['filename'])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source_path, target_path)

def main():
    args = parse_arguments()
    
    # Set parameters based on command line arguments
    corruption_type = args.corruption_type
    severity_level = args.severity_level
    plot_type = args.plot_type
    scene_name = args.scene_name
    output_base_dir = args.output_dir
    
    # Define source and output paths
    source_dataroot = f'/workspace/multicorrupt/{corruption_type}/{severity_level}'
    output_dir = os.path.join(output_base_dir, f'{corruption_type}_{severity_level}_{plot_type}_{scene_name}')
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=source_dataroot, verbose=False)
    
    # Get the scene index
    scene_index = get_scene_index_by_name(nusc, scene_name)
    if scene_index is None:
        print(f"Scene {scene_name} not found!")
        return
    
    print(f"The index for {scene_name} is {scene_index}")
    
    # Get the sample tokens in the scene
    my_scene = nusc.scene[scene_index]
    first_sample_token = my_scene['first_sample_token']
    current_sample_token = first_sample_token
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through samples in the scene and copy image files
    sample_count = 0
    while current_sample_token != '':
        print(f"Processing sample {sample_count + 1}...")
        copy_image_files_for_sample(nusc, current_sample_token, output_dir, plot_type)
        
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
        sample_count += 1
    
    print(f"Finished exporting {sample_count} samples from {scene_name} to {output_dir}")

if __name__ == "__main__":
    main()
