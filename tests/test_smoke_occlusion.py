import cv2
import numpy as np
import pytest

from nrec_utils.occlusions.smoke_occlusion import (
    Smoke,
    adjust_brightness_bisection,
    fake_depth_map,
)

H, W = 48, 64


@pytest.fixture
def image():
    """Mock BGR image: H x W x 3, uint8, random pixel values."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def mean_gray(img):
    return cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]


# ---------------------------------------------------------------- fake_depth_map


@pytest.mark.parametrize("shape", [(48, 64, 3), (37, 51, 3), (480, 640, 3), (48, 64)])
def test_depth_map_matches_input_height_and_width(shape):
    depth = fake_depth_map(np.zeros(shape, dtype=np.uint8))

    # depth is a single 2D map: same H and W as the input, no channels
    assert depth.shape == shape[:2]
    assert np.issubdtype(depth.dtype, np.floating)


def test_depth_map_default_values(image):
    depth = fake_depth_map(image)

    # values are clipped to [1, z_max] and smoothing cannot leave that range
    assert depth.min() >= 1.0
    assert depth.max() <= 300.0

    # depth only depends on the row, so every column is the same
    np.testing.assert_array_equal(depth, np.tile(depth[:, :1], (1, W)))

    # far at the top (sky / horizon), getting nearer towards the bottom
    assert np.all(np.diff(depth[:, 0]) <= 1e-9)

    # top rows are above the horizon and keep z_max
    assert depth[0, 0] == pytest.approx(300.0)

    # bottom row: cam_h / tan(tilt + fov/2) = 1.5 / tan(50deg) ~ 1.26,
    # lifted a little by the gaussian smoothing
    assert 1.0 <= depth[-1, 0] <= 1.5


def test_depth_map_custom_z_max(image):
    depth = fake_depth_map(image, z_max=50)

    assert depth.max() <= 50.0
    assert depth[0, 0] == pytest.approx(50.0)


# ---------------------------------------------------- adjust_brightness_bisection


def test_brightness_output_shape_dtype_and_range(image):
    result = adjust_brightness_bisection(image, factor=0.7)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


@pytest.mark.parametrize("factor", [0.85, 0.7, 0.55])
def test_brightness_reaches_target_factor(image, factor):
    result = adjust_brightness_bisection(image, factor=factor)

    assert mean_gray(result) / mean_gray(image) == pytest.approx(factor, abs=0.02)


def test_brightness_input_image_is_not_modified(image):
    original = image.copy()

    adjust_brightness_bisection(image, factor=0.7)

    np.testing.assert_array_equal(image, original)


# ------------------------------------------------------------------------- Smoke


@pytest.fixture
def occlusion():
    return Smoke(occ_factor=0.2)


def test_occ_name(occlusion):
    assert occlusion.get_occ_name() == "smoke"


@pytest.mark.parametrize("occ_factor", [0.1, 0.2, 0.3])
def test_output_shape_and_dtype(image, occ_factor):
    result = Smoke(occ_factor=occ_factor).generate_occluded_image(image)

    assert isinstance(result, np.ndarray)
    assert result.shape == (H, W, 3)
    assert result.dtype == np.uint8


def test_same_seed_gives_same_output(occlusion, image):
    first = occlusion.generate_occluded_image(image, seed=42)
    second = occlusion.generate_occluded_image(image, seed=42)

    np.testing.assert_array_equal(first, second)


def test_input_image_is_not_modified(occlusion, image):
    original = image.copy()

    occlusion.generate_occluded_image(image)

    np.testing.assert_array_equal(image, original)
