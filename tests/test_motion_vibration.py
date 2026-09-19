import numpy as np
import pytest

from nrec_utils.occlusions.motion_vibration import MotionVibration

H, W = 48, 64


@pytest.fixture
def image():
    """Mock RGB image: H x W x 3, uint8, random pixel values."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


@pytest.fixture
def occlusion():
    return MotionVibration(occ_factor=0.5)


def test_occ_factor(occlusion):
    assert occlusion.get_occ_factor() == 0.5


def test_occ_name(occlusion):
    assert occlusion.get_occ_name() == "motionVibration"


def test_output_has_same_shape_as_input(occlusion, image):
    result = occlusion.generate_occluded_image(image)

    assert isinstance(result, np.ndarray)
    assert result.shape == (H, W, 3)


def test_output_keeps_input_dtype(occlusion, image):
    result = occlusion.generate_occluded_image(image)

    assert result.dtype == image.dtype


def test_input_image_is_not_modified(occlusion, image):
    original = image.copy()

    occlusion.generate_occluded_image(image)

    np.testing.assert_array_equal(image, original)


def test_zero_occ_factor_returns_unchanged_image(image):
    # kernel size would round to 0; it is clamped to 1x1, which is an identity filter
    result = MotionVibration(occ_factor=0.0).generate_occluded_image(image)

    np.testing.assert_array_equal(result, image)
