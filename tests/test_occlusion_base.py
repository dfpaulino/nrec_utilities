import pytest

from nrec_utils.occlusions.occlusion_base import OcclusionBase


@pytest.fixture
def occlusion():
    return OcclusionBase(occ_factor=0.25, occ_name="blur")


def test_get_occ_factor(occlusion):
    assert occlusion.get_occ_factor() == 0.25


def test_get_occ_name(occlusion):
    assert occlusion.get_occ_name() == "blur"
