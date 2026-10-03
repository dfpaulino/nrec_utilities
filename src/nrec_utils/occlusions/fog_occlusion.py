
from scipy.ndimage import gaussian_filter1d
import numpy as np
import random
import cv2

from nrec_utils.occlusions.occlusion_base import OcclusionBase
#from nrec_utils.occlusions.utils.lime import LIME


# copied from adverse_weather article
def generate_fog_dummy(image, depth, visibility=None, fog_color=None):
    '''
    input:
        image - numpy array (h, w, c) 
        depth - numpy array (h, w)
    '''

    height, width  = depth.shape
    #perlin = perlin_noise(width, height, depth)

    depth_max = depth.max()
    
    if visibility:
        fog_visibility = visibility
    else:
        fog_visibility = float(np.random.randint(int(depth_max-0.2*depth_max), int(depth_max+0.2*depth_max)) )
        fog_visibility = np.clip(fog_visibility, 60, 200)

    VERTICLE_FOV = 60 #degrees
    CAMERA_ALTITUDE = 1.8 #meters
    VISIBILITY_RANGE_MOLECULE = 12  # m    12
    VISIBILITY_RANGE_AEROSOL = fog_visibility  # m     450
    ECM_ = 3.912 / VISIBILITY_RANGE_MOLECULE  # EXTINCTION_COEFFICIENT_MOLECULE /m
    ECA_ = 3.912 / VISIBILITY_RANGE_AEROSOL  # EXTINCTION_COEFFICIENT_AEROSOL /m


    FT = 70  # FOG_TOP m  31  70
    HT = 34  # HAZE_TOP m  300    34

    angle = np.repeat(-1*np.linspace(-0.5*VERTICLE_FOV, 0.5*VERTICLE_FOV, height).reshape(-1,1), axis=1, repeats=width)
    distance = depth / np.cos(np.radians(angle))
    elevation = CAMERA_ALTITUDE + distance * np.sin(np.radians(angle))

    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    ECA = ECA_
    c = 1 - elevation / (FT + 0.00001)
    c[c < 0] = 0
    #ECM = (ECM_ * c + (1 - c) * ECA_) * (perlin / 255)
    ECM = (ECM_ * c + (1 - c) * ECA_) 

    idx1 = np.logical_and(FT > elevation, elevation > HT)
    idx2 = elevation <= HT
    idx3 = elevation >= FT

    distance_through_haze[idx2] = distance[idx2]
    distance_through_fog[idx1] = (
        (elevation[idx1] - HT)
        * distance[idx1]
        / (elevation[idx1] - CAMERA_ALTITUDE)
    )
    distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
    distance_through_haze[idx3] = (
        (HT - CAMERA_ALTITUDE)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_fog[idx3] = (
        (FT - HT)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_haze_free[idx3] = (
        distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]
    )

    attenuation = np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)

    I_ex = image * attenuation[:,:,None]
    O_p = 1 - attenuation
    if fog_color is None:
        fog_color = np.random.randint(200,255)
    I_al = np.array([[[fog_color, fog_color, fog_color]]])

    I = I_ex + O_p[:,:,None] * I_al
    return I.astype(np.uint8)

# Not in use
def fake_depth_map_simple(image):
  H,W=image.shape[0],image.shape[1]

  FOV=60 # degreed Field of View on vertical

  CAMERA_H=1.50
  HORIZON_X=200 #pixl for horizon
  Z_MAX=35

  fy=int(np.round((H/2)/(np.tan(np.radians(FOV/2)))))

  x=np.arange(H)
  depth = np.full(H,Z_MAX,dtype=float)

  bellow_horizon = x>HORIZON_X # closeer
  above_horizon = x<HORIZON_X # further


  depth[bellow_horizon] = (
      fy * CAMERA_H /
      (x[bellow_horizon] - HORIZON_X)
  )

  depth = np.clip(depth, 2, Z_MAX)
  # here we have a step transition. above horizon we increase distance
  depth[above_horizon]=depth[above_horizon]*np.exp(0.01*(HORIZON_X-x[above_horizon]))

  # smoothen transition at horizon
  depth = gaussian_filter1d(
    depth,
    sigma=60
  )

  # depth is a 1D vector, we need a 2D, so repeat the value across the  columns W
  depth = np.tile(depth[:, None], (1, W))
  return depth

def fake_depth_map(image, fov_v=60, cam_h=1.5, tilt_deg=20, z_max=300, sigma_frac=0.03):
    """
    see calculation log book - calc horizon line30/09/2026
    """
    H, W = image.shape[:2]
    fy = (H / 2) / np.tan(np.radians(fov_v / 2)) # focal length
    cy = H / 2  # center of img

    theta = np.radians(tilt_deg)
    v_h = cy - fy * np.tan(theta)            # horizon row, derived from tilt
    #print(f'v_h {v_h}')

    v = np.arange(H, dtype=float)
    phi = theta + np.arctan((v - cy) / fy)   # ray angle below horizontal

    # see logbook 03/10/2026 for trig. calculations
    # rows below the horizon hit the ground (equivalent to phi > 0);
    # rows at/above it are sky/far background and keep z_max
    depth = np.full(H, z_max, dtype=float)
    ground = v > v_h + 1
    depth[ground] = cam_h / np.sin(phi[ground])   # distance along the ray to the ground
    depth = np.clip(depth, 1.0, z_max)

    # avoid the step, and use gaussian to smooth distance. 
    depth = gaussian_filter1d(depth, sigma=sigma_frac * H)
    # expand depth to a 2d. columns (W) are repeated
    return np.tile(depth[:, None], (1, W))

def getIlluminationMap_fromGray( img: np.ndarray) -> np.ndarray: 
    """
    Convert image to Grayscale
    """
    T = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    return T




class Fog(OcclusionBase):

    OCCLUSION_NAME='fogEffect'
    BETA_BASE=0.04
    MAX_VISIBILITY_Z = 50 # meters

    # define pxl intensity bins to check if scenario is dark or bright - values obtained by visual inspection!
    ILLUMINATION_BINS=[0,50,150,256]
    ILLUMINATION_TO_FOG = {0: (150, 180), 1: (210, 240), 2: (240, 255)}

    def __init__(self, occ_factor):
        #self._lime = LIME(iterations=25, alpha=1.0)
        super().__init__(occ_factor, Fog.OCCLUSION_NAME)

    #def getIlluminationMap(self, img: np.ndarray) -> np.ndarray: 
    #    self._lime.load(img)
    #    T = self._lime.illumMap()
    #    return T

    def generate_occluded_image(self,image):
       
       # occ_factor [0.1 0.2 0.3], betas from experimental view [0.01 0.02 0.03], and the last is really intense        
       beta = Fog.BETA_BASE*self._occ_factor*10

       # estimate brightness to calc fog colour
       #T = self.getIlluminationMap(img=image)
       T = getIlluminationMap_fromGray(image)
       hist,bins = np.histogram(T, bins=Fog.ILLUMINATION_BINS, range=(0,1))
       illumination_bin = hist.argmax()

       fog_color=random.randint(Fog.ILLUMINATION_TO_FOG[illumination_bin][0],Fog.ILLUMINATION_TO_FOG[illumination_bin][1])
       #print(f' Illum index is {illumination_bin}- {hist}')
       foggy_img = self._generate_fog(image=image,beta=beta,fog_color=fog_color)

       #depth=fake_depth_map(image=image)
       #foggy_img = generate_fog2(image=image,depth=depth,fog_color=fog_color)
       return foggy_img

    def _generate_fog(self,image,beta,fog_color):
  
        #depth= fake_depth_map_simple(image=image)
        depth= fake_depth_map(image=image,z_max=Fog.MAX_VISIBILITY_Z,sigma_frac=0.1)
        A = np.array([[[fog_color,fog_color,fog_color]]],dtype=np.uint8)

        #Light transmission through fog follows the Beer-Lambert law 
        transmission=np.exp(-beta*depth)
        foggy=np.round(image*(transmission[:,:,np.newaxis]) + (1-transmission[:,:,np.newaxis])*A).astype(np.uint8)
        foggy = np.clip(foggy, 0, 255,dtype=np.uint8)

        return foggy
