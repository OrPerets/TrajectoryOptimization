import numpy as np
import pytest

from src.utils import check_error_sanity, check_mapping_consistency


def test_check_error_sanity_nan():
    arr = np.array([0.0, np.nan])
    with pytest.raises(ValueError):
        check_error_sanity(arr, name="test errors")


def test_check_error_sanity_large():
    arr = np.array([1e6, 1e6])
    with pytest.raises(ValueError):
        check_error_sanity(arr, name="large errors", max_m=1e5)


def test_check_mapping_consistency_mismatch():
    radar = np.zeros((5, 3))
    telem = np.ones((5, 3)) * 2e4
    with pytest.raises(ValueError):
        check_mapping_consistency(radar, telem, threshold=1e3)
