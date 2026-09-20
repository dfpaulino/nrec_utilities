import pytest

from nrec_utils.occlusions import (
    MOTION_VIBRATION_OCCLUSION_NAME,
    OCCLUSION_REGISTRY,
    MotionVibration,
    OcclusionBase,
    available_occlusions,
    create_occlusion,
)


def test_motion_vibration_is_registered():
    assert MOTION_VIBRATION_OCCLUSION_NAME in available_occlusions()


@pytest.mark.parametrize("name", sorted(OCCLUSION_REGISTRY))
def test_registry_key_matches_the_instance_name(name):
    # the key is what the CLI accepts and the instance name is what the output
    # filename prefix is built from - if they drift, the two disagree silently
    assert create_occlusion(name, 0.5).get_occ_name() == name


@pytest.mark.parametrize("name", sorted(OCCLUSION_REGISTRY))
def test_registry_builds_a_configured_occlusion(name):
    occlusion = create_occlusion(name, 0.25)

    assert isinstance(occlusion, OcclusionBase)
    assert occlusion.get_occ_factor() == 0.25


@pytest.mark.parametrize("spelling", ["MOTION_VIBRATION", "motion-vibration",
                                      "motionvibration", "  motionVibration  "])
def test_aliases_resolve_to_the_canonical_occlusion(spelling):
    occlusion = create_occlusion(spelling, 0.5)

    assert isinstance(occlusion, MotionVibration)
    # whatever the user typed, the filename prefix stays canonical
    assert occlusion.get_occ_name() == MOTION_VIBRATION_OCCLUSION_NAME


def test_unknown_occlusion_lists_the_known_ones():
    with pytest.raises(ValueError, match=MOTION_VIBRATION_OCCLUSION_NAME):
        create_occlusion("gaussianSnow", 0.5)
