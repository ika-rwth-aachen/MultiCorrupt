import numpy as np
from scipy.ndimage import zoom as scizoom
import math
import cv2
import skimage as sk


seed = 1000
np.random.seed(seed)

""" brightness """
def brightness(x, severity):
    s = [.1,  .3, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + s, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + s, 0, 1)
        x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


""" motion blur """
def mb_gauss_function(x, mean, sigma):
    return (np.exp(- (x - mean)**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def mb_getMotionBlurKernel(width, sigma):
    k = mb_gauss_function(np.arange(width), 0, sigma*100)
    Z = np.sum(k)
    return k/Z

def mb_shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def imgmb(x, radius, sigma, angle):
    width = radius * 2 + 1
    kernel = mb_getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):

            break
        shifted = mb_shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

def img_motion_blur(x, severity):
    s = [0.06, 0.1, 0.13][severity - 1]
    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    x = imgmb(x, radius=15, sigma=s, angle=angle)
    return np.clip(x, 0, 255)




""" darkness """
def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def poisson_gaussian_noise(x, severity):
    s_poisson = 10 * [25, 12, 5][severity-1]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * s_poisson) / s_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08,  0.18, 0.26][severity-1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return np.uint8(x)

def low_light(x, severity):
    s = [0.60, 0.40, 0.30][severity-1]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, s, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity)
    return x_scaled


""" fog """
def plasma_fractal(mapsize=256, wibbledecay=1.7):
    """
    Generate a heightmap using diamond-square algorithm.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def noise(x, severity):
    s = [2.,3.,4.][severity - 1]
    attenuation_strength = 0.8
    backscatter_strength = 0.5
    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()
    
    # Generate the basic fog effect
    fog_effect = plasma_fractal(mapsize=map_size, wibbledecay=1.7)[:shape[0], :shape[1]]
    
    # Generate a spatially varying attenuation map and apply it to the fog effect
    attenuation_map = plasma_fractal(mapsize=map_size)[:shape[0], :shape[1]]
    fog_effect *= (1 - attenuation_strength * attenuation_map)
    
    # Generate a spatially varying backscatter map and apply it to the fog effect
    backscatter_map = plasma_fractal(mapsize=map_size)[:shape[0], :shape[1]]
    fog_effect *= (1 + backscatter_strength * backscatter_map)



    if len(shape) < 3 or shape[2] < 3:
        x += s * fog_effect
    else:
        x += s * fog_effect[..., np.newaxis]
        
    return np.clip(x * max_val / (max_val + s), 0, 1) * 255


def fog(x, severity):
    img = noise(x, severity)
    img_f = img / 255.0 
    (row, col, chs) = img.shape
    beta = [0.0005,0.001,0.002][severity - 1]
    A = 0.5
    size = math.sqrt(max(row, col))  
    y1, x1 = int(row * 0.65), int(col * 0.1)
    y2, x2 = int(row * 0.99), int(col * 0.85)
    for j in range(row):        
        for l in range(col):
            d_rect = math.sqrt(
                max(0, max(j - y2, y1 - j))**2 +
                max(0, max(l - x2, x1 - l))**2
            )
            
            d =  2.0* d_rect*(1+np.random.rand()/4) + size            
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td) 
    return img_f*255  




    
"""snow function"""
def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- (x - mean)**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z



def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img


def snow(x, severity=1):
    s = [(0.3,  5, 0.9, 7, 7, 0.9),
         (0.4,  4, 0.85, 8, 8, 0.85),
         (0.5,  3, 0.8, 10, 10, 0.8)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=s[0],
                                  scale=0.3)  
    
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], s[1])
    snow_layer[snow_layer < s[2]] = 0

    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

    snow_layer = _motion_blur(snow_layer, radius=s[3], sigma=s[4], angle=np.random.uniform(-135, -45))

    # The snow layer is rounded and cropped to the img dims
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]

    x = s[5] * x + (1 - s[5]) * np.maximum(x, cv2.cvtColor(x,
                                                            cv2.COLOR_RGB2GRAY).reshape(
        x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    
    # Adjust the brightness of the image (brightness factor is 0.9)
    brightness_factor = 0.9
    image = x * brightness_factor    
    
    try:
        return np.clip(image + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        print('ValueError for Snow, Exception handling')
        image[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(
            snow_layer, k=2)
        return np.clip(image, 0, 1) * 255