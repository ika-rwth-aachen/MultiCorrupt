import os
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Render corrupted nuScenes images and create a GIF animation.')
    parser.add_argument('-c', '--corruption_type', help='Corruption type', type=str, default='motionblur')
    parser.add_argument('-s', '--severity_level', help='Severity level (1, 2, 3)', type=str, default='1')
    parser.add_argument('-p', '--plot_type', help='Plot type (bev, front_camera, multi_view_camera)', type=str, default='multi_view_camera')
    parser.add_argument('-n', '--scene_name', help='Scene name', type=str, default='scene-0097')
    arguments = parser.parse_args()
    return arguments

args = parse_arguments()

# Set parameters based on command line arguments
corruption_type = args.corruption_type
severity_level = args.severity_level
plot_type = args.plot_type
scene_name = args.scene_name

nusc = NuScenes(version='v1.0-trainval', dataroot=f'/workspace/corrupted_nuscenes/robust-nuscenes/{corruption_type}/{severity_level}', verbose=False)


def get_scene_index_by_name(nusc, scene_name):
    for i, scene in enumerate(nusc.scene):
        if scene['name'] == scene_name:
            return i
    return None

def render_and_save_image(sample_token, output_path, plot_type):
    if plot_type == "front_camera":
        downscale_factor = 3
        nusc.render_pointcloud_in_image(
            sample_token,
            pointsensor_channel='LIDAR_TOP',
            camera_channel='CAM_FRONT',
            render_intensity=True,
            out_path=output_path
        )
    elif plot_type == "bev":
        downscale_factor = 4
        nusc.render_sample_data(
            nusc.get('sample', sample_token)['data']['LIDAR_TOP'],
            nsweeps=10,
            underlay_map=True,
            out_path=output_path
        )
        # Crop 300px from the top and 300px from the bottom
        img = Image.open(output_path)
        img = img.crop((0, 300, img.width, img.height - 300))
        img.save(output_path)
    elif plot_type == "multi_view_camera":
        downscale_factor = 8
        images = []
        camera_channels = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        for channel in camera_channels:
            temp_output_path = f"tmp/{channel}_{sample_token}.jpg"
            nusc.render_pointcloud_in_image(
                sample_token,
                pointsensor_channel='LIDAR_TOP',
                camera_channel=channel,
                render_intensity=True,
                out_path=temp_output_path
            )
            img = Image.open(temp_output_path)
            images.append(img)
            os.remove(temp_output_path)

        # Arrange images into a 2x3 grid
        total_width = max(img.width for img in images[:3]) * 3
        total_height = max(img.height for img in images[::2]) * 2
        new_image = Image.new('RGB', (total_width, total_height))

        x_offset = 0
        y_offset = 0
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
            if (i + 1) % 3 == 0:
                x_offset = 0
                y_offset += img.height

        new_image.save(output_path)
    
    # Resize
    img = Image.open(output_path)
    img = img.resize((img.width // downscale_factor, img.height // downscale_factor), Image.ANTIALIAS)
    img.save(output_path)
    plt.close()

# Test the function with a scene name.
scene_index = get_scene_index_by_name(nusc, scene_name)
print(f"The index for {scene_name} is {scene_index}")

# Get the sample tokens in the scene
my_scene = nusc.scene[scene_index]
first_sample_token = my_scene['first_sample_token']
current_sample_token = first_sample_token

image_paths = []  # List to store image paths

if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Iterate through samples in the scene, render point clouds, and save as JPG
while current_sample_token != '':
    output_path = f"tmp/image_{current_sample_token}.jpg"
    render_and_save_image(current_sample_token, output_path, plot_type)
    image_paths.append(output_path)

    current_sample = nusc.get('sample', current_sample_token)
    current_sample_token = current_sample['next']

# Combine saved images into a GIF animation
images = []
for image_path in image_paths:
    img = Image.open(image_path)
    images.append(img)

gif_path = f'{corruption_type}_{severity_level}_{plot_type}_{scene_name}_scene_animation.gif'
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)

# Remove temporary image files
for image_path in image_paths:
    os.remove(image_path)