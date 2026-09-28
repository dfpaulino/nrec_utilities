
from scipy.ndimage import gaussian_filter1d
import numpy as np
from nrec_utils.occlusions.occlusion_base import OcclusionBase

def fake_depth_map(image):
  H,W=image.shape[0],image.shape[1]

  FOV=60 # degreed Field of View on vertical

  CAMERA_H=1.50
  HORIZON_X=200 #pixl for horizon
  Z_MAX=35

  print(f'image dimensions H:{H} W:{W}')

  fx=int(np.round((H/2)/(np.tan(np.radians(FOV/2)))))

  x=np.arange(H)
  depth = np.full(H,Z_MAX,dtype=float)

  bellow_horizon = x>HORIZON_X # closeer
  above_horizon = x<HORIZON_X # further


  depth[bellow_horizon] = (
      fx * CAMERA_H /
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

  depth = np.tile(depth[:, None], (1, W))
  return depth

def generate_fog(image,beta):
  
  depth= fake_depth_map(image=image)
  fog_color=[[230,240,240]]
  A = np.array(fog_color,dtype=np.uint8)

  transmission=np.exp(-beta*depth)
  foggy=np.round(image*(transmission[:,:,np.newaxis]) + A*(1-transmission[:,:,np.newaxis])).astype(np.uint8)
  foggy = np.clip(foggy, 0, 255,dtype=np.uint8)

  return foggy

class Fog(OcclusionBase):

    OCCLUSION_NAME='fogEffect'
    BETA_BASE=0.01

    def __init__(self, occ_factor):
        super().__init__(occ_factor, Fog.OCCLUSION_NAME)

    def generate_occluded_image(self,image):
       
       # occ_factor [0.1 0.2 0.3], betas from experimental view [0.01 0.02 0.03], and the last is really intense        
       beta = Fog.BETA_BASE*self._occ_factor*10
       foggy_img = generate_fog(image=image,beta=beta)
       return foggy_img
