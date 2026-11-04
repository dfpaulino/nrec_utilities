
from scipy.ndimage import gaussian_filter1d,gaussian_filter
import numpy as np
import vnoise
import random
import cv2
import numpy.typing as npt
from typing import Union

from nrec_utils.occlusions.occlusion_base import OcclusionBase



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
    depth[ground] = cam_h / np.tan(phi[ground])   # distance along the ray to the ground
    depth = np.clip(depth, 1.0, z_max)

    # avoid the step, and use gaussian to smooth distance. 
    depth = gaussian_filter1d(depth, sigma=sigma_frac * H)
    # expand depth to a 2d. columns (W) are repeated
    depth_2d = np.tile(depth[:, None], (1, W)) 
    return depth_2d



def adjust_brightness_simple(image:np.ndarray, factor:float)->np.ndarray:
    """
    Adjust RGB image brightness for factor. Overall brightness is calulated 
    based on the grayscale pixel average. 
    INPUT:
        image: RGB np.ndarray
        factor: float factor [0-1] of intended brightness
            if factor is <1, then brightness is reduced eg factor=0.8 is 20% reduction
    """

    beta = -np.mean(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))*(1-factor)
    # cant use convertScaleAbs() as negative values can become positve
    # openCV applies saturate at the end
    new_img=cv2.addWeighted(image, factor, np.zeros_like(image), 0, beta)  
    return new_img


def adjust_brightness_bisection(image:np.ndarray,factor:float,interations:int=10):
    """
    Adjust RGB image brightness untill brigthness is factor of the original
    Original brightness is calculated from its grayscale.
     The image is converted to linear space see:
        Szeliski, R. (2022), Computer Vision: Algorithms and Applications,2nd ed., Springer.
        The chapter on image formation covers camera radiometric response and gamma.
    Returns the adjusted image on the sRGB space [0,255]
    """
    # approximation of sRGB to linear. Gamma correction
    image_lin = (image/255.0)**2.2
    
    gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target = factor * cv2.mean(gray_before)[0]
    # Find the illumination scale that gives the target brightness (bisection)

    lo,hi=0.0,1.0
    for _ in range(interations):
        k = (lo+hi)/2
        #invert back to sRGB, re-encode
        out_sRGB=np.clip((image_lin*k)**(1/2.2),0.0,1.0)
        out_sRGB = (out_sRGB*255).astype(np.uint8)

        if cv2.mean(cv2.cvtColor(out_sRGB, cv2.COLOR_BGR2GRAY))[0] > target:
            hi=k
        else:
            lo=k     

    return out_sRGB  


class Smoke(OcclusionBase):

    OCCLUSION_NAME='smoke'
    BETA_BASE=0.06
    MAX_VISIBILITY_Z = 50 # meters

    # eyball adjusted tbh!
    TRANSMISSION_SIGMA_FACTOR=0.5

    def __init__(self, occ_factor):
        super().__init__(occ_factor, Smoke.OCCLUSION_NAME)


    def generate_occluded_image(self,image,bboxs:list=[],seed:int=42):
       """
       Overrides base method
       """
       
       # occ_factor [0.1 0.2 0.3], betas from experimental observation       
       beta = Smoke.BETA_BASE*self._occ_factor*10
       # create the fake depth
       depth= fake_depth_map(image=image,z_max=Smoke.MAX_VISIBILITY_Z,sigma_frac=0.1)

       smoke2d = self.generate_random_smoke2d(image,seed=seed)

       # some contol on smoke density 1.5 is part of the control
       # occ 0.1 0.2 0.2 -> 15% 30% 45 density
       smoke_density = (self._occ_factor*1.6)*smoke2d/255.0
       transmission = self.calculate_transmission_gaussian(beta=beta,depth=depth,density=smoke_density)
       

       # darken the image and add smoke based on the transmission factor steps of 15% for occ [0.1 0.2 0.3]
       bright_pc=1.0-(0.15*10*self._occ_factor) #85% 70% 55% occ [0.1 0.2 0.3]

       # work on images in linear space to have better control
       #image_darken = adjust_brightness_simple(image=image,factor=bright_pc)
       image_darken=adjust_brightness_bisection(image=image,factor=bright_pc)

       #gamma correction approximation
       image_lin=(image_darken/255.0)**2.2

       smoke3d = cv2.merge([smoke2d,smoke2d,smoke2d])
       #gamma correction approximation
       smoke3d_lin = (smoke3d/255.0)**2.2
       # linear operation
       smoky=image_lin*(transmission[:,:,np.newaxis]) + (1-transmission[:,:,np.newaxis])*smoke3d_lin
    
       # From above, the transmission at low distance is very high, removing most of the smoke effect at low distance.
       # However its needed as with depthness smoke will restrict visibility..similar to fog. But we also need 
       #some smoke effect at low distance (0.1 0.2 0.3 -> blend 10-20-30%)
       alpha_blend=1-self._occ_factor
       beta_blend = (1-alpha_blend)
       smoky=cv2.addWeighted(smoky, alpha_blend, smoke3d_lin, beta_blend, 0.0)

       # re-encode back to sRGB
       smoky_sRGB = (np.clip(smoky**(1/2.2), 0.0, 1.0)*255)

       return smoky_sRGB.astype(np.uint8)

    def generate_random_smoke2d(self,image:np.ndarray,seed:int=42)->np.ndarray:
        """
        Generate a 2D smoke cloud based on perlin noise
        Input 
            image used for shape
            seed: int used as seed for random generation
        """
        shape=image.shape
        # things to tune
        scale=300
        octaves=3
        persistence=.5
        lacunarity = 1.5

        noise = vnoise.Noise()
        y=np.linspace(0,shape[0]/scale,shape[0])
        x=np.linspace(0,shape[1]/scale,shape[1])
        
        smoke2d = noise.noise2(y,x,grid_mode=True,octaves=octaves,persistence=persistence,lacunarity=lacunarity,base=seed)

        smoke2d=((smoke2d-np.min(smoke2d))/(np.max(smoke2d)-np.min(smoke2d))*255).astype("uint8")

        #smoke2d_gaus = cv2.GaussianBlur(smoke2d,(1,1),0)
        return smoke2d


    
    def calculate_transmission_gaussian(self, beta,depth,density: Union[float,npt.NDArray],bboxs=[]):
        """
        Calculates Transmission based on Beer-Lambert law. May take BBox to ensure the transmission is the 
        same within the BBox area, eg person standing has same transmission along the rows.
        density is used as a factor. for example smoke will affect transmission, denser smoke less transmission.
        """
        transmission=np.exp(-beta*depth*density)
        H,W=transmission.shape

        # set transmission as same value as ground IF there are BBOXs
        for bbox in bboxs:
            x_c, y_c = int(bbox[0]*W), int(bbox[1]*H)
            # give extra 10% of box for smoothen
            w_half, h_half = int(1.0*bbox[2]*W/2), int(1.0*bbox[3]*H/2)
            y0, y1 = max(y_c-h_half,0), min(y_c+h_half,H)
            x0, x1 = max(x_c-w_half,0), min(x_c+w_half,W)
        
            # transmission at the object's feet or waist?(clamped to the image)
            target_transmission = transmission[min(y_c+h_half,H-1), min(x_c,W-1)]
            transmission[y0:y1,x0:x1]=target_transmission
            #there is a big edge, so we need to smoothen
            sigma =Smoke.TRANSMISSION_SIGMA_FACTOR*min(w_half,h_half)+1
            transmission=gaussian_filter(transmission,sigma=sigma)

        return transmission
