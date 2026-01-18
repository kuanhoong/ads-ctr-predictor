
import cv2
import numpy as np

def get_brightness_features(img_rgb):
    """
    1.2: Brightness (YUV and HSL)
    """
    # YUV
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0]
    
    # HLS (L channel)
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    l_channel = img_hls[:, :, 1]
    
    features = []
    for channel in [y_channel, l_channel]:
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(np.max(channel))
        features.append(np.min(channel))
        
    return features

def get_saturation_features(img_rgb):
    """
    1.2: Saturation (HSL/HSV)
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_channel = img_hsv[:, :, 1]
    
    features = []
    features.append(np.mean(s_channel))
    features.append(np.std(s_channel))
    features.append(np.max(s_channel))
    features.append(np.min(s_channel))
    return features

def get_colorfulness_feature(img_rgb):
    """
    1.2: Colorfulness (RGB-based calculation)
    Hasler & Suesstrunk metric.
    """
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)
    
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return [colorfulness]

def get_naturalness_feature(img_rgb):
    """
    1.2: Naturalness (skin, grass, sky detection)
    Returns sum of proportions.
    """
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    H = img_hls[:, :, 0] # 0-179
    L = img_hls[:, :, 1] # 0-255
    S = img_hls[:, :, 2] # 0-255
    
    # Thresholds mapped from 0-100/0-1 to 0-255
    # L: 20-80 -> 51-204
    # S: >0.1 -> >25
    mask_valid = (L >= 51) & (L <= 204) & (S >= 25)
    
    # Skin (Approx Red/Orange): 0-25, 165-180
    mask_skin = ((H <= 25) | (H >= 165)) & mask_valid
    
    # Grass (Green): 35-85
    mask_grass = (H >= 35) & (H <= 85) & mask_valid
    
    # Sky (Blue): 90-130
    mask_sky = (H >= 90) & (H <= 130) & mask_valid
    
    total = img_rgb.shape[0] * img_rgb.shape[1]
    prop_skin = np.sum(mask_skin) / total
    prop_grass = np.sum(mask_grass) / total
    prop_sky = np.sum(mask_sky) / total
    
    naturalness = prop_skin + prop_grass + prop_sky
    return [naturalness]

def get_grayscale_features(img_rgb):
    """
    1.2: Grayscale features (std dev)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return [np.std(gray)]

def extract_global_features(img_rgb):
    f_bright = get_brightness_features(img_rgb)
    f_sat = get_saturation_features(img_rgb)
    f_col = get_colorfulness_feature(img_rgb)
    f_nat = get_naturalness_feature(img_rgb)
    f_gray = get_grayscale_features(img_rgb)
    
    return np.concatenate([f_bright, f_sat, f_col, f_nat, f_gray])
