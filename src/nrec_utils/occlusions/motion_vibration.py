
import numpy as np
import cv2
from nrec_utils.occlusions.occlusion_base import OcclusionBase



class MotionVibration(OcclusionBase):
    
    INTERNAL_FACTOR=1
    OCCLUSION_NAME='motionVibration'

    def __init__(self, occ_factor):
        super().__init__(occ_factor, MotionVibration.OCCLUSION_NAME)

    def generate_occluded_image(self,image):
        # based on min image shape and INTERNAL_FACTOR (educated guess) will provide the kernel's size
        # bigger size more vibration induced
        # at least 1: a 0-size kernel makes cv2.filter2D fail (1x1 kernel returns the image unchanged)
        
        kernel_vibration = self._generate_cross_kernel_weight_decay(self,image)
        # convolve filter on image
        return cv2.filter2D(image,-1, kernel_vibration,borderType=cv2.BORDER_DEFAULT)


    def _generate_cross_kernel_weight_decay(self,image):

        filter_shape = max(1, int(round(min(image.shape[0], image.shape[1]) * self.get_occ_factor() * MotionVibration.INTERNAL_FACTOR)))

        #filters should be odd size
        if filter_shape%2==0:
            filter_shape+=1
        
        # create CROSS kernel and notmalize it
        kernel_morph =cv2.getStructuringElement(cv2.MORPH_CROSS,(filter_shape,filter_shape))
        # filter normalization
        kernel_vibration = kernel_morph/np.sum(kernel_morph)
        kernel_morph=kernel_morph.astype(float)

        # apply weight decay over the filter. pixl influence reduces as it distances from the center.
        # apply  1 / (1+distance)  or use some sort of e^(-beta*distance)
        # we skip the center as its 1
        center_filter=filter_shape//2
        for i in range(center_filter +1,filter_shape):
            dist_from_center= np.abs(i-center_filter)
            weight = 1/(1+dist_from_center)
            kernel_morph[center_filter,i]=kernel_morph[center_filter,i]*weight
            kernel_morph[i,center_filter]=kernel_morph[i,center_filter]*weight

            kernel_morph[center_filter,(filter_shape-1-i)]=kernel_morph[center_filter,(filter_shape-1-i)]*weight
            kernel_morph[(filter_shape-1-i),center_filter]=kernel_morph[(filter_shape-1-i),center_filter]*weight
            #print(f'idx {(filter_shape-1-i)}')
        
        # filter normalization
        kernel_vibration = kernel_morph/np.sum(kernel_morph)

        return kernel_vibration
