
from nrec_utils.occlusions.occlusion_base import OcclusionBase
from nrec_utils.occlusions.motion_vibration import MotionVibration

# The canonical name of an occlusion is what its instances report from
# get_occ_name(), because that is what ends up as the output filename prefix.
# The registry is keyed by that same string so the two can never drift apart.
MOTION_VIBRATION_OCCLUSION_NAME = 'motionVibration'

OCCLUSION_REGISTRY = {
    MOTION_VIBRATION_OCCLUSION_NAME: MotionVibration,
}

OCCLUSION_REGISTRY_LIST = [MotionVibration]

def available_occlusions() -> list:
    """The canonical names of every registered occlusion, sorted."""
    return sorted(OCCLUSION_REGISTRY)


def _normalise(name: str) -> str:
    return name.strip().replace("_", "").replace("-", "").casefold()


def create_occlusion(name: str, occ_factor: float) -> OcclusionBase:
    """
    Build the occlusion registered under `name`, at `occ_factor`.

    Matching ignores case, underscores and hyphens, so 'MOTION_VIBRATION',
    'motion-vibration' and 'motionVibration' all resolve to the same class.
    The instance still reports the canonical spelling, so an alias can never
    produce a second set of output filenames.
    """
    lookup = {_normalise(key): key for key in OCCLUSION_REGISTRY}
    canonical = lookup.get(_normalise(name))
    if canonical is None:
        raise ValueError(
            f"unknown occlusion '{name}'; known: {', '.join(available_occlusions())}"
        )
    return OCCLUSION_REGISTRY[canonical](occ_factor)


def create_occlusions_instances(occ_factor:float) -> OcclusionBase:
    instances=[]
    for occ in OCCLUSION_REGISTRY_LIST:
        instances.append(occ(occ_factor=occ_factor))
    return instances

__all__ = [
    "OcclusionBase",
    "MotionVibration",
    "MOTION_VIBRATION_OCCLUSION_NAME",
    "OCCLUSION_REGISTRY",
    "OCCLUSION_REGISTRY_LIST"
    "available_occlusions",
    "create_occlusion",
    "create_occlusions_instances"
]
