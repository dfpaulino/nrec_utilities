class OcclusionBase:
    """
    Base class for an image degradation ("occlusion") applied to a dataset.

    Every implementation so far is PIXEL-ONLY: it changes pixel values but
    moves no geometry, so the bounding boxes of the source image stay valid and
    create_occluded_set copies the YOLO label file verbatim. That generator
    checks the invariant at runtime (the output must keep the input's height
    and width) rather than trusting it.

    Adding a GEOMETRIC occlusion - a crop, rotation, warp, or a pasted occluder
    that hides an annotated object - breaks that assumption, and the labels
    would have to be transformed alongside the pixels. Do not bolt a separate
    transform_labels() method onto this class to do that: such an occlusion
    derives the label transform from the same (usually randomised) parameters
    it used on the pixels, so two calls would force the class to stash state
    between them, which is how datasets end up quietly mislabelled. Replace
    generate_occluded_image with a single

        apply(self, image, label_lines) -> (image, label_lines)

    and migrate the existing subclasses, which is a small change while there
    are only a handful of them.
    """

    def __init__(self,occ_factor:float,occ_name:str):
        print (f'INFO - creating occlusion instance of {occ_name}')
        self._occ_factor=occ_factor
        self._occ_name=occ_name
        

    def get_occ_factor(self) ->float:
        return self._occ_factor

    def get_occ_name(self) ->str:
        """Canonical name, used as the output filename prefix and registry key."""
        return self._occ_name

    def generate_occluded_image(self, image):
        """Return a degraded copy of `image`, with its height and width unchanged."""
        pass
