"""
This file contains tests for background_corrections.py.
@author: Nathanael Jöhrmann
"""
import numpy as np
import pytest

from afm_tools.background_correction import BGCorrectionType, correct_background

# -------------------------------------------------------------------------------------------------------------
# --------------------------------------- immutable testcases ------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# zero plane - should stay unchanged for all BG corrections!
zero_plane = ((0, 0, 0),
              (0, 0, 0),
              (0, 0, 0))

# offset plane - only offset - corrections should change nothing for keep_z_offset == True, else return zero_plane.
offset_plane = ((1., 1., 1.),
                (1., 1., 1.),
                (1., 1., 1.))

x_tilted_plane = ((0, 0, 0),
                  (1, 1, 1),
                  (2, 2, 2))

y_tilted_plane = ((0, 1, 2),
                  (0, 1, 2),
                  (0, 1, 2))

curved_plane = ((1, 0, 1),
                (1, 0, 1),
                (1, 0, 1))


class TestBackgroundCorrectionTypes:
    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType])
    def test_zero_plane(self, correction):
        result = correct_background(np.array(zero_plane), correction, keep_offset=False)
        assert np.all(np.isclose(result, zero_plane))

        correct_background(np.array(zero_plane), correction, keep_offset=True)
        assert np.all(np.isclose(result, zero_plane))

    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType if not c == BGCorrectionType.raw_data])
    def test_offset_plane(self, correction):
        result = correct_background(np.array(offset_plane), correction, keep_offset=True)
        assert np.all(np.isclose(result, offset_plane))

        result = correct_background(np.array(offset_plane), correction, keep_offset=False)
        assert np.all(np.isclose(result, zero_plane))

    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType if not c == BGCorrectionType.raw_data])
    def test_x_tilted_plane(self, correction):
        result = correct_background(np.array(x_tilted_plane), correction, keep_offset=True)
        if correction is BGCorrectionType.legendre_0:
            assert np.all(np.isclose(result, x_tilted_plane))
        else:
            assert np.all(np.isclose(result, offset_plane))

        result = correct_background(np.array(x_tilted_plane), correction, keep_offset=False)
        if correction is BGCorrectionType.legendre_0:
            assert np.all(np.isclose(result, (x_tilted_plane - np.mean(x_tilted_plane))))
        else:
            assert np.all(np.isclose(result, zero_plane))

    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType if not c == BGCorrectionType.raw_data])
    def test_y_tilted_plane(self, correction):
        result = correct_background(np.array(y_tilted_plane), correction, keep_offset=True)
        if correction is BGCorrectionType.legendre_0:
            assert np.all(np.isclose(result, y_tilted_plane))
        else:
            assert np.all(np.isclose(result, offset_plane))

        result = correct_background(np.array(y_tilted_plane), correction, keep_offset=False)
        if correction is BGCorrectionType.legendre_0:
            assert np.all(np.isclose(result, (y_tilted_plane - np.mean(y_tilted_plane))))
        else:
            assert np.all(np.isclose(result, zero_plane))

    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType if not c == BGCorrectionType.raw_data])
    def test_curved_plane(self, correction):
        result = correct_background(np.array(curved_plane), correction, keep_offset=True)
        if correction in [BGCorrectionType.legendre_0, BGCorrectionType.legendre_1, BGCorrectionType.gradient]:
            assert np.all(np.isclose(result, curved_plane))
        else:
            assert np.all(np.isclose(result, (zero_plane + np.mean(curved_plane))))

        result = correct_background(np.array(curved_plane), correction, keep_offset=False)
        if correction in [BGCorrectionType.legendre_0, BGCorrectionType.legendre_1, BGCorrectionType.gradient]:
            assert np.all(np.isclose(result, (curved_plane - np.mean(curved_plane))))
        else:
            assert np.all(np.isclose(result, zero_plane))

    @pytest.mark.parametrize("correction", [c for c in BGCorrectionType])
    def test_none(self, correction):
        result = correct_background(None, correction, keep_offset=False)
        assert result is None
