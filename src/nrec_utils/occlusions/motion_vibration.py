
import numpy as np
import cv2
from nrec_utils.occlusions.occlusion_base import OcclusionBase



class MotionVibration(OcclusionBase):
    
    INTERNAL_FACTOR=0.25
    OCCLUSION_NAME='motionVibration'

    def __init__(self, occ_factor):
        super().__init__(occ_factor, MotionVibration.OCCLUSION_NAME)

    def generate_occluded_image(self,image):
        # based on min image shape and INTERNAL_FACTOR (educated guess) will provide the kernel's size
        # bigger size more vibration induced
        # at least 1: a 0-size kernel makes cv2.filter2D fail (1x1 kernel returns the image unchanged)
        filter_shape = max(1, int(round(min(image.shape[0], image.shape[1]) * self.get_occ_factor() * MotionVibration.INTERNAL_FACTOR)))
        #create CROSS kernel and notmalize it
        kernel_morph =cv2.getStructuringElement(cv2.MORPH_CROSS,(filter_shape,filter_shape))
        # filter normalization
        kernel_vibration = kernel_morph/np.sum(kernel_morph)

        # convolve filter on image
        return cv2.filter2D(image,-1, kernel_vibration)

