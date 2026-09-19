class OcclusionBase:

    def __init__(self,occ_factor:float,occ_name:str):
        self._occ_factor=occ_factor
        self._occ_name=occ_name

    def get_occ_factor(self) ->float:
        return self._occ_factor
    def get_occ_name(self) ->str:
        return self._occ_name
    
    def generate_occluded_image(self, image):
        pass
