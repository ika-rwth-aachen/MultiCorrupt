import os
import copy
import math
from typing import Dict, List, Tuple
from copy import deepcopy
import numpy as np
import pickle5 as pickle
from pathlib import Path
import yaml
import itertools
from scipy.constants import speed_of_light as c     # in m/s
from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PI = np.pi

seed = 1000
np.random.seed(seed)
RNG = np.random.default_rng(seed)

""" motion blur """
def pts_motion(points, severity):
    s = [0.06, 0.1, 0.13][severity - 1]
    
    trans_std = [s, s, s]
    noise_translate = np.array([
    np.random.normal(0, trans_std[0], 1),
    np.random.normal(0, trans_std[1], 1),
    np.random.normal(0, trans_std[2], 1),
    ]).T
    
    points[:, 0:3] += noise_translate
    num_points = points.shape[0]
    jitters_x = np.clip(np.random.normal(loc=0.0, scale=trans_std[0]*0.15, size=num_points), -3 * trans_std[0], 3 * trans_std[0])
    jitters_y = np.clip(np.random.normal(loc=0.0, scale=trans_std[1]*0.2, size=num_points), -3 * trans_std[1], 3 * trans_std[1])
    jitters_z = np.clip(np.random.normal(loc=0.0, scale=trans_std[2]*0.12, size=num_points), -3 * trans_std[2], 3 * trans_std[2])

    points[:, 0] += jitters_x
    points[:, 1] += jitters_y
    points[:, 2] += jitters_z
    return points


""" spatial misalignment """
def transform_points(points, severity):
    """
    Rotate and translate a set of points.
    
    Parameters:
    points (numpy.ndarray): A 2D array where each row represents a point (x, y, z, ...).
    angle_degree (float): The rotation angle in degrees.
    
    Returns:
    numpy.ndarray: The transformed points.
    """
    s = [(0.2, 1), (0.4, 2), (0.6, 3)][severity - 1]
    
    
    # Convert the angle from degrees to radians
    rand_num = np.random.rand()
    
    # Check if the random number is less than the given probability
    if rand_num < s[0]:
        
        # Convert the angle from degrees to radians
        theta = np.radians(s[1])

        # Define the rotation matrix for rotation around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the y-axis
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the z-axis
        rotation_matrix_z = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Combine the three rotations
        rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

                
        # Extract the x, y, z coordinates of the points
        xyz_points = points[:, :3]
        
        # Apply the rotation matrix to the x, y, z coordinates
        rotated_xyz = np.dot(xyz_points, rotation_matrix.T)
        
        # Define the translation vector (2 units along the x-axis)
        translation_vector = np.array([2, 0, 0])
        
        # Apply the translation to the rotated x, y, z coordinates
        translated_xyz = rotated_xyz + translation_vector
        
        # Concatenate the translated x, y, z coordinates with the other properties of the points
        transformed_points = np.hstack((translated_xyz, points[:, 3:]))
        
        return transformed_points
    
    else:
        # If the random number is not less than the given probability, return the original points
        return points


""" beam reduce """
def reduce_LiDAR_beamsV2(pts, severity):
    s = [16, 8, 4][severity - 1]
    
    if s == 16:
        allowed_beams = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    elif s == 8:
        allowed_beams = [1, 5, 9, 13, 17, 21, 25, 29]
    elif s == 4:
        allowed_beams = [1, 9, 17, 25]
    
    mask = np.full(pts.shape[0], False)
    for beam in allowed_beams:
        beam_mask = pts[:, 4] == beam
        mask = np.logical_or(beam_mask, mask)
    return pts[mask, :]


""" points missing """
def pointsreducing(pts, severity):
    """
    Simulates missing lidar points based on a given severity level.

    Args:
    pts: A numpy array of lidar points.
    severity: An integer between 1 and 3, where 1 is the least severe and 3 is the most severe.

    Returns:
    A numpy array of lidar points with missing points.
    """
    s = [70, 80, 90][severity - 1]

    size = pts.shape[0]
    nr_of_samps = int(round(size * ((100 - s) / 100)))  # Calculate number of points to keep

    permutations = np.random.permutation(size)  # Generate random permutation of indices
    ind = permutations[:nr_of_samps]  # Select indices for points to keep

    pts = pts[ind]  # Extract points based on selected indices

    return pts


'''  fog function'''
INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / "fog_lookup_tables" 

class ParameterSet:

    def __init__(self, **kwargs) -> None:

        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.1                                    # in m              (displacement of transmitter and receiver)
        self.ROH_T = 0.01                               # in m              (radius of the transmitter aperture)
        self.ROH_R = 0.01                               # in m              (radius of the receiver aperture)
        self.GAMMA_T_DEG = 2                            # in deg            (opening angle of the transmitter's FOV)
        self.GAMMA_R_DEG = 3.5                          # in deg            (opening angle of the receiver's FOV)
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)


        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_integral_dict(p: ParameterSet) -> Dict:
    alpha =p.alpha
    beta=p.beta
    filename = INTEGRAL_PATH / f'integral_0m_to_200m_stepsize_0.1m_alpha_{alpha}_beta_{beta}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)

    return integral_dict


def P_R_fog_hard(p: ParameterSet, pc: np.ndarray) -> np.ndarray:
    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)
    pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])
    return pc


def P_R_fog_soft(p: ParameterSet, pc: np.ndarray, original_intesity: np.ndarray,  noise: int, gain: bool = False,
                 noise_variant: str = 'v1') -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = np.zeros(pc.shape)
    fog_mask = np.zeros(len(pc), dtype=bool)

    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)

    min_fog_response = np.inf
    max_fog_response = 0
    num_fog_responses = 0

    integral_dict = get_integral_dict(p)

    r_noise = RNG.integers(low=1, high=20, size=1)[0]
    r_noise = 10
    for i, r_0 in enumerate(r_zeros):

        # load integral values from precomputed dict
        key = float(str(round(r_0, 1)))
        # limit key to a maximum of 200 m
        fog_distance, fog_response = integral_dict[min(key, 200)]
        fog_response = fog_response * original_intesity[i] * (r_0 ** 2) * p.beta / p.beta_0

        # limit to 255
        # fog_response = min(fog_response, 255)

        if fog_response > pc[i, 3]:

            fog_mask[i] = 1

            num_fog_responses += 1

            scaling_factor = fog_distance / r_0

            augmented_pc[i, 0] = pc[i, 0] * scaling_factor
            augmented_pc[i, 1] = pc[i, 1] * scaling_factor
            augmented_pc[i, 2] = pc[i, 2] * scaling_factor
            augmented_pc[i, 3] = fog_response

            # keep 5th feature if it exists
            if pc.shape[1] > 4:
                augmented_pc[i, 4] = pc[i, 4]

            if noise > 0:

                if noise_variant == 'v1':

                    # add uniform noise based on initial distance
                    distance_noise = RNG.uniform(low=r_0 - noise, high=r_0 + noise, size=1)[0]
                    noise_factor = r_0 / distance_noise

                elif noise_variant == 'v2':

                    # add noise in the power domain
                    power = RNG.uniform(low=-1, high=1, size=1)[0]
                    noise_factor = max(1.0, noise/5) ** power       # noise=10 => noise_factor ranges from 1/2 to 2

                elif noise_variant == 'v3':

                    # add noise in the power domain
                    power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                    noise_factor = max(1.0, noise*4/10) ** power    # noise=10 => ranges from 1/2 to 4

                elif noise_variant == 'v4':

                    additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]
                    new_dist = fog_distance + additive
                    noise_factor = new_dist / fog_distance

                else:

                    raise NotImplementedError(f"noise variant '{noise_variant}' is not implemented (yet)")

                augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor
                augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor
                augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor

            if fog_response > max_fog_response:
                max_fog_response = fog_response

            if fog_response < min_fog_response:
                min_fog_response = fog_response

        else:
            augmented_pc[i] = pc[i]

    if gain:
        max_intensity = np.ceil(max(augmented_pc[:, 3]))
        gain_factor = 255 / max_intensity
        augmented_pc[:, 3] *= gain_factor

    simulated_fog_pc = None
    num_fog = 0
    if num_fog_responses > 0:
        fog_points = augmented_pc[fog_mask]
        simulated_fog_pc = fog_points
        num_fog = len(fog_points)


    info_dict = {'min_fog_response': min_fog_response,
                 'max_fog_response': max_fog_response,
                 'num_fog_responses': num_fog_responses,}

    return augmented_pc, simulated_fog_pc,  num_fog, info_dict


def simulate_fog(severity, pc: np.ndarray, noise: int, gain: bool = False, noise_variant: str = 'v1',
                 hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    s = [(0.02,0.008) ,(0.03,0.008),(0.06,0.05)][severity - 1]
    p = ParameterSet(alpha=s[0],beta=s[1]) 
    augmented_pc = copy.deepcopy(pc)
    original_intensity = copy.deepcopy(pc[:, 3])

    info_dict = None
    simulated_fog_pc = None

    if hard:
        augmented_pc = P_R_fog_hard(p, augmented_pc)
    if soft:
        augmented_pc, simulated_fog_pc,  num_fog, info_dict = P_R_fog_soft(p, augmented_pc, original_intensity,  noise, gain,
                                                                 noise_variant)

    return augmented_pc, simulated_fog_pc, num_fog, info_dict


'''  snow function'''
def estimate_ground_plane(point_cloud):
    min_ground = -1.3 
    max_ground = -2.1
    # Assuming point_cloud is a numpy array with shape (N, 3) representing (x, y, z) coordinates

    valid_loc = (point_cloud[:, 2] < min_ground) & (point_cloud[:, 2] > max_ground)
    point_cloud = point_cloud[valid_loc]

    if len(point_cloud) < 25:
        w = [0, 0, -1]
        h = -1.85
        print("Not enought points. Use default flat world assumption!!")
    else:
        # Create RANSACRegressor model
        model = make_pipeline(StandardScaler(), RANSACRegressor())

        # Fit the model to the data
        model.fit(point_cloud[:, :2], point_cloud[:, 2])

        # Extract the estimated coefficients (slope and intercept)
        w = np.zeros(3)
        w[0] = model.named_steps['ransacregressor'].estimator_.coef_[0]
        w[1] = model.named_steps['ransacregressor'].estimator_.coef_[1]
        w[2] = -1.0
        w = w / np.linalg.norm(w)
        h = model.named_steps['ransacregressor'].estimator_.intercept_
    
    if h < max_ground or h > min_ground:
        w = [0, 0, -1]
        h = -1.85
        print("Bad RANSAC Parameters. Use default flat world assumption!")
        
    return w, h

def calculate_plane(pointcloud, standart_height=-1.55):
    """
    caluclates plane from loaded pointcloud
    returns the plane normal w and lidar height h.
    :param pointcloud: binary with x,y,z, coordinates
    :return:           w, h
    """

    # Filter points which are close to ground based on mounting position
    valid_loc = (pointcloud[:, 2] < -1.55) & \
                (pointcloud[:, 2] > -1.86 - 0.01 * pointcloud[:, 0]) & \
                (pointcloud[:, 0] > 10) & \
                (pointcloud[:, 0] < 70) & \
                (pointcloud[:, 1] > -3) & \
                (pointcloud[:, 1] < 3)
    pc_rect = pointcloud[valid_loc]

    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, -1]
        # Standard height from vehicle mounting position
        h = standart_height
    else:
        try:
            reg = RANSACRegressor(loss='squared_loss', max_trials=1000).fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = -1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)

        except:
            # If error occurs fall back to flat earth assumption
            print('Was not able to estimate a ground plane. Using default flat earth assumption')
            w = [0.0, 0.0, -1.0]
            # Standard height from vehicle mounting position
            h = standart_height
    
    # if estimated h is not reasonable fall back to flat earth assumption
    if abs(h - standart_height) > 1.5:
        print('Estimated h is not reasonable. Using default flat earth assumption')
        w = [0.0, 0.0, -1.0]
        h = standart_height
    
    return w, h

with open(os.path.join(Path(__file__).parent.absolute(), "nuscenes_snow.yaml"), 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

def compute_occupancy(snowfall_rate: float, terminal_velocity: float, snow_density: float=0.1) -> float:
    """
    :param snowfall_rate:           Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:       Varies from 0.2 to 2                                    [m/s]
    :param snow_density:            Varies from 0.01 to 0.2 depending on snow wetness       [g/cm³]
    :return:                        Occupancy ratio.
    """
    water_density = 1.0

    return (water_density * snowfall_rate) / ((3.6 * 10 ** 6) * (snow_density * terminal_velocity))

def snowfall_rate_to_rainfall_rate(snowfall_rate: float, terminal_velocity: float,
                                   snowflake_density: float = 0.1, snowflake_diameter: float = 0.003) -> float:
    """
    :param snowfall_rate:       Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:   Varies from 0.2 to 2                                    [m/s]
    :param snowflake_density:   Varies from 0.01 to 0.2 depending on snow wetness       [g/cm^3]
    :param snowflake_diameter:  Varies from 1 to 10                                     [m]

    :return:
    rainfall_rate:              Varies from 0.5 (slight rain) to 50 (violent shower)    [mm/h]
    """

    rainfall_rate = np.sqrt((snowfall_rate / (487 * snowflake_density * snowflake_diameter * terminal_velocity))**3)

    return rainfall_rate

def ransac_polyfit(x, y, order=3, n=15, k=100, t=0.1, d=15, f=0.8):
    # Applied https://en.wikipedia.org/wiki/Random_sample_consensus
    # Taken from https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c
    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    bestfit = np.polyfit(x, y, order)
    besterr = np.sum(np.abs(np.polyval(bestfit, x) - y))
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit

def estimate_laser_parameters(pointcloud_planes, calculated_indicent_angle, power_factor=15, noise_floor=0.7,
                              debug=True, estimation_method='linear'):
    """
    :param pointcloud_planes: Get all points which correspond to the ground
    :param calculated_indicent_angle: The calculated incident angle for each individual point
    :param power_factor: Determines, how much more Power is available compared to a groundplane reflection.
    :param noise_floor: What are the minimum intensities that could be registered
    :param debug: Show additional Method
    :param estimation_method: Method to fit to outputted laser power.
    :return: Fits the laser outputted power level and noiselevel for each point based on the assumed ground floor reflectivities.
    """
    # normalize intensitities
    normalized_intensitites = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle)
    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    # linear model
    p = None
    stat_values = None
    if len(normalized_intensitites) < 3:
        return None, None, None, None
    if estimation_method == 'linear':
        reg = linregress(distance, normalized_intensitites)
        w = reg[0]
        h = reg[1]
        p = [w, h]
        stat_values = reg[2:]
        relative_output_intensity = power_factor * (p[0] * distance + p[1])

    elif estimation_method == 'poly':
        # polynomial 2degre fit
        p = np.polyfit(np.linalg.norm(pointcloud_planes[:, :3], axis=1),
                       normalized_intensitites, 2)
        relative_output_intensity = power_factor * (
                p[0] * distance ** 2 + p[1] * distance + p[2])


    # estimate minimum noise level therefore get minimum reflected intensitites
    hist, xedges, yedges = np.histogram2d(distance, normalized_intensitites, bins=(50, 2555),
                                          range=((10, 70), (5, np.abs(np.max(normalized_intensitites)))))
    idx = np.where(hist == 0)
    hist[idx] = len(pointcloud_planes)
    ymins = np.argpartition(hist, 2, axis=1)[:, 0]
    min_vals = yedges[ymins]
    idx = np.where(min_vals > 5)
    min_vals = min_vals[idx]
    idx1 = [i + 1 for i in idx]
    x = (xedges[idx] + xedges[idx1]) / 2

    if estimation_method == 'poly':
        pmin = ransac_polyfit(x, min_vals, order=2)
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance ** 2 + pmin[1] * distance + pmin[2])
    elif estimation_method == 'linear':
        if len(min_vals) > 3:
            pmin = linregress(x, min_vals)
        else:
            pmin = p
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance + pmin[1])
    # Guess that noise level should be half the road relfection


    return relative_output_intensity, adaptive_noise_threshold, p, stat_values

def process_single_channel(root_path: str, particle_file_prefix: str, orig_pc: np.ndarray, beam_divergence: float,
                           order: List[int], channel_infos: List, channel: int) -> Tuple:
    """
    :param root_path:               Needed for training on GPU cluster.
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param orig_pc:                 N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param beam_divergence:         Equivalent to the total beam opening angle (in degree).
    :param order:                   Order of the particle disks.
    :param channel_infos            List of Dicts containing sensor calibration info.

    :param channel:                 Number of the LiDAR channel [0, 63].

    :return:                        Tuple of
                                    - intensity_diff_sum,
                                    - idx,
                                    - the augmented points of the current LiDAR channel.
    """
    
    intensity_diff_sum = 0

    index = order[channel]

    min_intensity = 0  #channel_infos[channel].get('min_intensity', 0)  # not all channels contain this info

    focal_distance = channel_infos[channel]['focal_distance'] * 100
    focal_slope = channel_infos[channel]['focal_slope']
    focal_offset = (1 - focal_distance / 13100) ** 2                # from velodyne manual

    particle_file = f'{particle_file_prefix}_{index + 1}.npy'

    channel_mask = orig_pc[:, 4] == channel
    idx = np.where(channel_mask == True)[0]

    pc = orig_pc[channel_mask]
    # print(pc.shape)

    N = pc.shape[0]

    x, y, z, intensity = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    distance = np.linalg.norm([x, y, z], axis=0)

    center_angles = np.arctan2(y, x)  # in range [-PI, PI]
    center_angles[center_angles < 0] = center_angles[center_angles < 0] + 2 * PI  # in range [0, 2*PI]

    beam_angles = -np.ones((N, 2))

    beam_angles[:, 0] = center_angles - np.radians(beam_divergence / 2)  # could lead to negative values
    beam_angles[:, 1] = center_angles + np.radians(beam_divergence / 2)  # could lead to values above 2*PI

    # put beam_angles back in range [0, 2*PI]
    beam_angles[beam_angles < 0] = beam_angles[beam_angles < 0] + 2 * PI
    beam_angles[beam_angles > 2 * PI] = beam_angles[beam_angles > 2 * PI] - 2 * PI

    occlusion_list = get_occlusions(beam_angles=beam_angles, ranges_orig=distance, beam_divergence=beam_divergence,
                                    root_path=root_path, particle_file=particle_file)

    lidar_range = 120                       # in meter
    intervals_per_meter = 10                # => 10cm discretization
    beta_0 = 1 * 10 ** -6 / PI
    tau_h = 1e-8                            #  value 10ns taken from HDL64-S1 specsheet

    M = lidar_range * intervals_per_meter

    M_extended = int(np.ceil(M + c * tau_h * intervals_per_meter))
    lidar_range_extended = lidar_range + c * tau_h

    R = np.round(np.linspace(0, lidar_range_extended, M_extended), len(str(intervals_per_meter)))

    for j, beam_dict in enumerate(occlusion_list):

        d_orig = distance[j]
        i_orig = intensity[j]

        if channel in [53, 55, 56, 58]:
            max_intensity = 230
        else:
            max_intensity = 255

        i_adjusted = i_orig - 255 * focal_slope * np.abs(focal_offset - (1 - d_orig/120)**2)
        i_adjusted = np.clip(i_adjusted, 0, max_intensity)      # to make sure we don't get negative values

        CA_P0 = i_adjusted * d_orig ** 2 / beta_0

        if len(beam_dict.keys()) > 1:                           # otherwise there is no snowflake in the current beam

            i = np.zeros(M_extended)

            for key, tuple_value in beam_dict.items():

                if key != -1:                                   # if snowflake
                    i_orig = 0.9 * max_intensity                # set i to 90% reflectivity
                    CA_P0 = i_orig / beta_0                     # and do NOT normalize with original range

                r_j, ratio = tuple_value

                start_index = int(np.ceil(r_j * intervals_per_meter))
                end_index = int(np.floor((r_j + c * tau_h) * intervals_per_meter) + 1)

                for k in range(start_index, end_index):
                    i[k] += received_power(CA_P0, beta_0, ratio, R[k], r_j, tau_h)

            max_index = np.argmax(i)
            i_max = i[max_index]
            d_max = (max_index / intervals_per_meter) - (c * tau_h / 2)

            i_max += max_intensity * focal_slope * np.abs(focal_offset - (1 - d_max/120)**2)
            i_max = np.clip(i_max, min_intensity, max_intensity)

            if abs(d_max - d_orig) < 2 * (1 / intervals_per_meter):  # only alter intensity

                pc[j, 4] = 1

                new_i = int(i_max)

                intensity_diff_sum += i_orig - new_i

            else:  # replace point of hard target with snowflake

                pc[j, 4] = 2

                d_scaling_factor = d_max / d_orig

                pc[j, 0] = pc[j, 0] * d_scaling_factor
                pc[j, 1] = pc[j, 1] * d_scaling_factor
                pc[j, 2] = pc[j, 2] * d_scaling_factor

                new_i = int(i_max)

            assert new_i >= 0, f'new intensity is negative ({new_i})'

            clipped_i = np.clip(new_i, min_intensity, max_intensity)

            pc[j, 3] = clipped_i

        else:
            pc[j, 4] = 0

    return intensity_diff_sum, idx, pc


def binary_angle_search(angles: List[float], low: int, high: int, angle: float) -> int:
    """
    Adapted from https://www.geeksforgeeks.org/python-program-for-binary-search

    :param angles:                  List of individual endpoint angles.
    :param low:                     Start index.
    :param high:                    End index.
    :param angle:                   Query angle.

    :return:                        Index of angle if present in list of angles, else -1
    """

    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If angle is present at the middle itself
        if angles[mid] == angle:
            return mid

        # If angle is smaller than mid, then it can only be present in left sublist
        elif angles[mid] > angle:
            return binary_angle_search(angles, low, mid - 1, angle)

        # Else the angle can only be present in right sublist
        else:
            return binary_angle_search(angles, mid + 1, high, angle)

    else:
        # Angle is not present in the list
        return -1


def compute_occlusion_dict(beam_angles: Tuple[float, float], intervals: np.ndarray,
                           current_range: float, beam_divergence: float) -> Dict:
    """
    :param beam_angles:         Tuple of angles (left, right).
    :param intervals:           N-by-3 array of particle tangent angles and particle distance from origin.
    :param current_range:       Range to the original hard target.
    :param beam_divergence:     Equivalent to the total beam opening angle (in degree).

    :return:
    occlusion_dict:             Dict containing a tuple of the distance and the occluded angle by respective particle.
                                e.g.
                                0: (distance to particle, occlusion ratio [occluded angle / total angle])
                                1: (distance to particle, occlusion ratio [occluded angle / total angle])
                                3: (distance to particle, occlusion ratio [occluded angle / total angle])
                                7: (distance to particle, occlusion ratio [occluded angle / total angle])
                                ...
                                -1: (distance to original target, unocclusion ratio [unoccluded angle / total angle])

                                all (un)occlusion ratios always sum up to the value 1
    """

    try:
        N = intervals.shape[0]
    except IndexError:
        N = 1

    right_angle, left_angle = beam_angles

    # Make everything properly sorted in the corner case of phase discontinuity.
    if right_angle > left_angle:
        right_angle = right_angle - 2*PI
        right_left_order_violated = intervals[:, 0] > intervals[:, 1]
        intervals[right_left_order_violated, 0] = intervals[right_left_order_violated, 0] - 2*PI

    endpoints = sorted(set([right_angle] + list(itertools.chain(*intervals[:, :2])) + [left_angle]))
    diffs = np.diff(endpoints)
    n_intervals = diffs.shape[0]

    assignment = -np.ones(n_intervals)

    occlusion_dict = {}

    for j in range(N):

        a1, a2, distance = intervals[j]

        i1 = binary_angle_search(endpoints, 0, len(endpoints), a1)
        i2 = binary_angle_search(endpoints, 0, len(endpoints), a2)

        assignment_made = False

        for k in range(i1, i2):

            if assignment[k] == -1:
                assignment[k] = j
                assignment_made = True

        if assignment_made:
            ratio = diffs[assignment == j].sum() / np.radians(beam_divergence)
            occlusion_dict[j] = (distance, np.clip(ratio, 0, 1))

    ratio = diffs[assignment == -1].sum() / np.radians(beam_divergence)
    occlusion_dict[-1] = (current_range, np.clip(ratio, 0, 1))

    return occlusion_dict

def tangent_angles_to_interval_angles(angles: np.ndarray, right_angle: float, left_angle: float,
                                      right_angle_hit: np.ndarray, left_angle_hit: np.ndarray) -> np.ndarray:
    """
    :param angles:              N-by-2 array containing tangent angles.
    :param right_angle:         Right beam angle.
    :param left_angle:          Left beam angle.
    :param right_angle_hit:     Flag if right beam angle has been exceeded.
    :param left_angle_hit:      Flag if left beam angle has been exceeded.

    :return:                    N-by-2 array of corrected tangent angles that do not exceed beam limits.
    """

    angles[right_angle_hit, 0] = right_angle
    angles[left_angle_hit, 1] = left_angle

    return angles

def do_angles_intersect_particles(angles: np.ndarray, particle_centers: np.ndarray) -> np.ndarray:
    """
    Assumption: either the ray that corresponds to an angle or its opposite ray intersects with all particles.

    :param angles:              (M,) array of angles in the range [0, 2*PI).
    :param particle_centers:    (N, 2) array of particle centers (abscissa, ordinate).

    :return:
    """
    try:
        M = angles.shape[0]
    except IndexError:
        M = 1

    try:
        N = particle_centers.shape[0]
    except IndexError:
        N = 1

    x, y = particle_centers[:, 0], particle_centers[:, 1]

    angle_to_centers = np.arctan2(y, x)
    angle_to_centers[angle_to_centers < 0] = angle_to_centers[angle_to_centers < 0] + 2*PI                      # (N, 1)

    angle_differences = np.tile(angles, (1, N)) - np.tile(angle_to_centers.T, (M, 1))                           # (M, N)

    answer = np.logical_or.reduce((np.abs(angle_differences) < PI/2,
                                   np.abs(angle_differences - 2*PI) < PI/2,
                                   np.abs(angle_differences + 2*PI) < PI/2))                                    # (M, N)

    return answer

def angles_to_lines(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param angles:              M-by-2 array of angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :return:
    a_s:                        N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    b_s:                        N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    """

    tan_directions = np.tan(angles)                                                                             # (M, 2)

    directions_vertical = np.logical_or(angles == PI/2, angles == 3 * PI/2)
    directions_not_vertical = np.logical_not(directions_vertical)

    a_s = np.zeros_like(angles)
    b_s = np.zeros_like(angles)

    a_s[np.where(directions_vertical)] = 1
    b_s[np.where(directions_vertical)] = 0

    a_s[np.where(directions_not_vertical)] = -tan_directions[np.where(directions_not_vertical)]
    b_s[np.where(directions_not_vertical)] = 1

    # a_s[np.abs(a_s) < EPSILON] = 0              # to prevent -0 value

    return a_s, b_s




def tangent_lines_to_tangent_angles(lines: Tuple[np.ndarray, np.ndarray], center_angles: np.ndarray) -> np.ndarray:
    """
    :param lines:               Tuple of two N-by-2 arrays holding the $a$ and $b$ coefficients of the tangents.
    :param center_angles:       N-by-1 array containing the angle to the particle center.

    :return:
    angles:                     N-by-2 array of tangent angles (right angle first, left angle second).
    """

    a_s, b_s = lines

    try:
        N = center_angles.shape[0]
    except IndexError:
        N = 1

    angles = -np.ones((N, 2))                                                                                   # (N, 2)

    ray_1_angles = np.arctan(-a_s/b_s)                                      # in range [-PI/2, PI/2]            # (N, 2)
    ray_2_angles = deepcopy(ray_1_angles) + PI                              # in range [PI/2, 3*PI/2]           # (N, 2)

    # correct value range
    ray_1_angles[ray_1_angles < 0] = ray_1_angles[ray_1_angles < 0] + 2*PI  # in range [0, 2*PI]                # (N, 2)
    ray_1_angles = np.abs(ray_1_angles)                                     # to prevent -0 value

    # catch special case if line is vertical
    ray_1_angles[b_s == 0] = PI/2
    ray_2_angles[b_s == 0] = 3*PI/2

    tangent_1_angles = np.column_stack((ray_1_angles[:, 0], ray_2_angles[:, 0]))                                # (N, 2)
    tangent_2_angles = np.column_stack((ray_1_angles[:, 1], ray_2_angles[:, 1]))                                # (N, 2)

    for i, tangent_angles in enumerate([tangent_1_angles, tangent_2_angles]):

        tangent_difference = tangent_angles - np.column_stack((center_angles, center_angles))                   # (N, 2)

        correct_ray = np.logical_or.reduce((np.abs(tangent_difference) < PI/2,
                                            np.abs(tangent_difference - 2*PI) < PI/2,
                                            np.abs(tangent_difference + 2*PI) < PI/2))                          # (N, 2)

        angles[:, i] = tangent_angles[np.where(correct_ray)]                                                    # (N, 2)

    angles.sort(axis=1)

    # swap order where discontinuity is crossed
    swap = angles[:, 1] - angles[:, 0] > PI
    angles[swap, 0], angles[swap, 1] = angles[swap, 1], angles[swap, 0]

    return angles

def tangents_from_origin(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param samples:             N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    :return:
    a:                          N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    b:                          N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    """

    # Solve systems of equations that encode the following information:
    # 1) rays include origin,
    # 2) rays are tangent to the circles corresponding to the particles, i.e., they intersect with the circles at
    # exactly one point.

    x_s, y_s, r_s = samples[:, 0], samples[:, 1], samples[:, 2]

    try:
        N = samples.shape[0]
    except IndexError:
        N = 1

    discriminants = r_s * np.sqrt(x_s ** 2 + y_s ** 2 - r_s ** 2)

    case_1 = np.abs(x_s) - r_s == 0  # One of the two lines is vertical.
    case_2 = np.logical_not(case_1)  # Both lines are not vertical.

    a_1_case_1, b_1_case_1 = np.ones(N), np.zeros(N)
    a_2_case_1, b_2_case_1 = (y_s ** 2 - x_s ** 2) / (2 * x_s * y_s), - np.ones(N)

    a_1_case_2 = (-x_s * y_s + discriminants) / (r_s ** 2 - x_s ** 2)
    a_2_case_2 = (-x_s * y_s - discriminants) / (r_s ** 2 - x_s ** 2)
    b_1_case_2 = -np.ones(N)
    b_2_case_2 = -np.ones(N)

    # Compute the coefficients by distinguishing the two cases.
    a_1, a_2, b_1, b_2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    a_1[case_1] = a_1_case_1[case_1]
    a_2[case_1] = a_2_case_1[case_1]
    b_1[case_1] = b_1_case_1[case_1]
    b_2[case_1] = b_2_case_1[case_1]

    a_1[case_2] = a_1_case_2[case_2]
    a_2[case_2] = a_2_case_2[case_2]
    b_1[case_2] = b_1_case_2[case_2]
    b_2[case_2] = b_2_case_2[case_2]

    a = np.column_stack((a_1, a_2))
    b = np.column_stack((b_1, b_2))

    return a, b

def distances_of_points_to_lines(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    :param points:      N-by-2 array of points, where each row contains the coordinates (abscissa, ordinate) of a point
    :param a:           M-by-1 array of $a$ coefficients of lines
    :param b:           M-by-1 array of $b$ coefficients of lines
    :param c:           M-by-1 array of $c$ coefficients of lines
                        where ax + by = c

    :return:            N-by-M array containing distances of points to lines
    """

    try:
        N = points.shape[0]
    except IndexError:
        N = 1

    abscissa, ordinate = points[:, 0, np.newaxis], points[:, 1, np.newaxis]

    numerators = np.dot(abscissa, a.T) + np.dot(ordinate, b.T) + np.dot(np.ones((N, 1)), c.T)

    denominators = np.dot(np.ones((N, 1)), np.sqrt(a ** 2 + b ** 2).T)

    return np.abs(numerators / denominators)


def get_occlusions(beam_angles: np.ndarray, ranges_orig: np.ndarray, root_path: str, particle_file: str,
                   beam_divergence: float) -> List:
    """
    :param beam_angles:         M-by-2 array of beam endpoint angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :param ranges_orig:         M-by-1 array of original ranges corresponding to beams (in m).
    :param root_path:           Needed for training on GPU cluster.

    :param particle_file:       Path to N-by-3 array of all sampled particles as disks,
                                where each row contains abscissa and ordinate of the disk center and disk radius (in m).
    :param beam_divergence:     Equivalent to the opening angle of an individual LiDAR beam (in degree).

    :return:
    occlusion_list:             List of M Dicts.
                                Each Dict contains a Tuple of
                                If key == -1:
                                - distance to the original hard target
                                - angle that is not occluded by any particle
                                Else:
                                - the distance to an occluding particle
                                - the occluded angle by this particle

    """

    M = np.shape(beam_angles)[0]
    # print("M shape is :()".format(M))

    if root_path:
        path = Path(root_path) / 'training' / 'snowflakes' / 'npy' / particle_file
    else:
        path = Path(__file__).parent.absolute() / 'npy' / particle_file

    all_particles = np.load(str(path))
    x, y, _ = all_particles[:, 0], all_particles[:, 1], all_particles[:, 2]

    all_particle_ranges = np.linalg.norm([x, y], axis=0)                                                        # (N,)
    all_beam_limits_a, all_beam_limits_b = angles_to_lines(beam_angles)                                       # (M, 2)

    occlusion_list = []

    # Main loop over beams.
    for i in range(M):

        current_range = ranges_orig[i]                                                                          # (K,)

        right_angle = beam_angles[i, 0]
        left_angle = beam_angles[i, 1]

        in_range = np.where(all_particle_ranges < current_range)

        particles = all_particles[in_range]                                                                     # (K, 3)

        x, y, particle_radii = particles[:, 0], particles[:, 1], particles[:, 2]

        particle_angles = np.arctan2(y, x)                                                                      # (K,)
        particle_angles[particle_angles < 0] = particle_angles[particle_angles < 0] + 2 * PI

        tangents_a, tangents_b = tangents_from_origin(particles)                                              # (K, 2)

        ################################################################################################################
        # Determine whether centers of the particles lie inside the current beam,
        # which is first sufficient condition for intersection.
        standard_case = np.logical_and(right_angle <= particle_angles, particle_angles <= left_angle)
        seldom_case = np.logical_and.reduce((right_angle - 2 * PI <= particle_angles, particle_angles <= left_angle,
                                             np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))
        seldom_case_2 = np.logical_and.reduce((right_angle <= particle_angles, particle_angles <= left_angle + 2 * PI,
                                               np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))

        center_in_beam = np.logical_or.reduce((standard_case, seldom_case, seldom_case_2))  # (K,)
        ################################################################################################################

        ################################################################################################################
        # Determine whether distances from particle centers to beam rays are smaller than the radii of the particles,
        # which is second sufficient condition for intersection.
        beam_limits_a = all_beam_limits_a[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_b = all_beam_limits_b[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_c = np.zeros((2, 1))  # origin                                                              # (2, 1)

        # Get particle distances to right and left beam limit.
        distances = distances_of_points_to_lines(particles[:, :2],
                                                   beam_limits_a, beam_limits_b, beam_limits_c)                 # (K, 2)

        radii_intersecting = distances < np.column_stack((particle_radii, particle_radii))                      # (K, 2)

        intersect_right_ray = do_angles_intersect_particles(right_angle, particles[:, 0:2]).T                 # (K, 1)
        intersect_left_ray = do_angles_intersect_particles(left_angle, particles[:, 0:2]).T                   # (K, 1)

        right_beam_limit_hit = np.logical_and(radii_intersecting[:, 0], intersect_right_ray[:, 0])
        left_beam_limit_hit = np.logical_and(radii_intersecting[:, 1], intersect_left_ray[:, 0])

        ################################################################################################################
        # Determine whether particles intersect the current beam by taking the disjunction of the above conditions.
        particles_intersect_beam = np.logical_or.reduce((center_in_beam,
                                                         right_beam_limit_hit, left_beam_limit_hit))            # (K,)

        ################################################################################################################

        intersecting_beam = np.where(particles_intersect_beam)

        particles = particles[intersecting_beam]  # (L, 3)
        particle_angles = particle_angles[intersecting_beam]
        tangents_a = tangents_a[intersecting_beam]
        tangents_b = tangents_b[intersecting_beam]
        tangents = (tangents_a, tangents_b)
        right_beam_limit_hit = right_beam_limit_hit[intersecting_beam]
        left_beam_limit_hit = left_beam_limit_hit[intersecting_beam]

        # Get the interval angles from the tangents.
        tangent_angles = tangent_lines_to_tangent_angles(tangents, particle_angles)                           # (L, 2)

        # Correct tangent angles that do exceed beam limits.
        interval_angles = tangent_angles_to_interval_angles(tangent_angles, right_angle, left_angle,
                                                              right_beam_limit_hit, left_beam_limit_hit)        # (L, 2)

        ################################################################################################################
        # Sort interval angles by increasing distance from origin.
        distances_to_origin = np.linalg.norm(particles[:, :2], axis=1)                                          # (L,)

        intervals = np.column_stack((interval_angles, distances_to_origin))                                     # (L, 3)
        ind = np.argsort(intervals[:, -1])
        intervals = intervals[ind]                                                                              # (L, 3)

        occlusion_list.append(compute_occlusion_dict((right_angle, left_angle),
                                                     intervals,
                                                     current_range,
                                                     beam_divergence))

    return occlusion_list


def simulate_snow(pc: np.ndarray,
                  label: np.ndarray,
                  severity: int,
                  beam_divergence: float,
                  shuffle: bool=True,
                  noise_floor: float=0.7,
                  root_path: str=None) -> Tuple:
    """
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param show_progressbar:        Flag if tqdm should display a progessbar.
    :param only_camera_fov:         Flag if the camera field of view (FOV) filter should be applied.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        Tuple of
                                    - Tuple of the following statistics
                                        - num_attenuated,
                                        - avg_intensity_diff
                                    - N-by-4 array of the augmented pointcloud.
    """

    assert pc.shape[1] == 5

    labels = copy.deepcopy(label).reshape(-1)
    labels = np.vectorize(learning_map.__getitem__)(labels)
    driveable_surface = labels == 11
    other_flat = labels == 12
    sidewalk = labels == 13

    mask1 = np.logical_or(driveable_surface, other_flat)
    ground = np.logical_or(mask1, sidewalk)

    pc_ground = pc[ground, :]
    calculated_indicent_angle = np.arccos(-np.divide(np.matmul(pc_ground[:, :3], np.asarray([0, 0, 1])),
                                                    np.linalg.norm(pc_ground[:, :3],
                                                                axis=1) * np.linalg.norm([0, 0, 1])))
    
    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    sensor_info = Path(__file__).parent.resolve() / 'snow_laser.yaml'

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']
    # num_channels = 32

    channels = range(num_channels)
    order = list(range(num_channels))
    
    if shuffle:
        np.random.shuffle(order)

    channel_list = [None] * num_channels

    s = [(0.5,1.2), (2.5,1.6), (1.5,0.4)][severity - 1]
    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    particle_file_prefix = f'gunn_{rain_rate}_{occupancy}' 
    
    channel_list = []

    for channel in channels:
        result = process_single_channel(root_path, particle_file_prefix,
                                        orig_pc, beam_divergence, order,
                                        channel_infos, channel)
        channel_list.append(result)


    intensity_diff_sum = 0
    # import pdb; pdb.set_trace()
    for item in channel_list:

        tmp_intensity_diff_sum, idx, pc_ = item

        intensity_diff_sum += tmp_intensity_diff_sum

        aug_pc[idx] = pc_

    aug_pc[:, 3] = aug_pc[:, 3] 
    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)
    num_removed = np.logical_not(scattered_or_above_threshold).sum()

    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    stats = num_attenuated, num_removed, avg_intensity_diff

    return  stats, aug_pc, ground



def simulate_snow_sweep(pc: np.ndarray,
                        severity: int,
                        beam_divergence: float,
                        shuffle: bool=True,
                        noise_floor: float=0.7,
                        root_path: str=None) -> Tuple:
    """
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param show_progressbar:        Flag if tqdm should display a progessbar.
    :param only_camera_fov:         Flag if the camera field of view (FOV) filter should be applied.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        Tuple of
                                    - Tuple of the following statistics
                                        - num_attenuated,
                                        - avg_intensity_diff
                                    - N-by-4 array of the augmented pointcloud.
    """

    assert pc.shape[1] == 5

    w, h = estimate_ground_plane(pc)
    ground = np.logical_and(np.matmul(pc[:, :3], np.asarray(w)) + h < 0.5,
                            np.matmul(pc[:, :3], np.asarray(w)) + h > -0.5)

    pc_ground = pc[ground, :]
    calculated_indicent_angle = np.arccos(-np.divide(np.matmul(pc_ground[:, :3], np.asarray([0, 0, 1])),
                                                    np.linalg.norm(pc_ground[:, :3],
                                                                axis=1) * np.linalg.norm([0, 0, 1])))
    
    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    sensor_info = Path(__file__).parent.resolve() / 'snow_laser.yaml'

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']
    # num_channels = 32

    channels = range(num_channels)
    order = list(range(num_channels))
    
    if shuffle:
        np.random.shuffle(order)

    channel_list = [None] * num_channels

    s = [(0.5,1.2), (2.5,1.6), (1.5,0.4)][severity - 1]
    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    particle_file_prefix = f'gunn_{rain_rate}_{occupancy}' 
    
    channel_list = []

    for channel in channels:
        result = process_single_channel(root_path, particle_file_prefix,
                                        orig_pc, beam_divergence, order,
                                        channel_infos, channel)
        channel_list.append(result)


    intensity_diff_sum = 0

    for item in channel_list:

        tmp_intensity_diff_sum, idx, pc_ = item

        intensity_diff_sum += tmp_intensity_diff_sum

        aug_pc[idx] = pc_

    aug_pc[:, 3] = aug_pc[:, 3] 
    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)
    num_removed = np.logical_not(scattered_or_above_threshold).sum()

    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    stats = num_attenuated, num_removed, avg_intensity_diff

    return  stats, aug_pc


def received_power(CA_P0: float, beta_0: float, ratio: float, r: float, r_j: float, tau_h: float) -> float:
    answer = ((CA_P0 * beta_0 * ratio * xsi(r_j)) / (r_j ** 2)) * np.sin((PI * (r - r_j)) / (c * tau_h)) ** 2
    return answer

def xsi(R: float, R_1: float = 0.9, R_2: float = 1.0) -> float:
    if R <= R_1:    # emitted ligth beam from the tansmitter is not captured by the receiver
        return 0
    elif R >= R_2:  # emitted ligth beam from the tansmitter is fully captured by the receiver
        return 1
    else:           # emitted ligth beam from the tansmitter is partly captured by the receiver
        m = (1 - 0) / (R_2 - R_1)
        b = 0 - (m * R_1)
        y = m * R + b
        return y