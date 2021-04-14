"""
This module contains functions for background correction. The different types are available via Enum BGCorrectionType.
@author: Nathanael Jöhrmann
"""
from enum import Enum, auto
from functools import partial
from typing import Optional

import numpy as np
from numpy.polynomial import Legendre


def subtract_mean_level(array2d: np.ndarray) -> Optional[np.ndarray]:
    """
    Correct an offset in the array2d by subtracting the mean level.
    :param array2d:
    :return: ndarray
    """
    if array2d is None:
        return None

    result = array2d - array2d.mean()
    return result


def subtract_legendre_fit(array2d: np.ndarray, keep_offset: bool = False, deg: int = 1) -> Optional[np.ndarray]:
    """
    Use a legendre polynomial fit of degree legendre_deg in X and Y direction to correct background.
    legendre_deg = 0 ... subtract mean value
    legendre_deg = 1 ... subtract mean plane
    legendre_deg = 2 ... subtract simple curved mean surface
    legendre_deg = 3 ... also corrects "s-shaped" distortion
    ...
    """
    if array2d is None:
        return None
    if deg == 0 and keep_offset:
        return array2d.copy()  # return a copy of input data
    n_row = np.linspace(-1, 1, array2d.shape[0])
    n_col = np.linspace(-1, 1, array2d.shape[1])
    mean_row = array2d.mean(axis=1)
    mean_col = array2d.mean(axis=0)

    fit_x = Legendre.fit(n_row, mean_row, deg)
    fit_y = Legendre.fit(n_col, mean_col, deg)

    result = array2d.copy()
    result = (result.transpose() - np.polynomial.legendre.legval(n_row, fit_x.coef)).transpose()
    result = result - np.polynomial.legendre.legval(n_col, fit_y.coef)
    if keep_offset:
        result = result + 2 * array2d.mean()  # mean was subtracted 2 times (once for fit_x ans once for fit_y)
    else:
        result = result + array2d.mean()
    return result


def subtract_mean_gradient_plane(array2d: np.ndarray, keep_offset: bool = False) -> Optional[np.ndarray]:
    """
    Returns 2d numpy.ndarray with subtracted mean gradient plane from given array2d. Using the gradient might give
     better results, when the measurement has asymmetric structures like large objects on a surface.
                                  ____________________
    example: ____________________|                   |__
    """
    if array2d is None:
        return None
    # result = array2d[:] # !!! slicing of np.ndarray only cretes viw, not copy !!!
    result = array2d.copy()  # !!! slicing of np.ndarray only cretes viw, not copy !!!

    try:
        value_gradient = np.gradient(array2d)
    except ValueError:
        print("ValueError in subtract_mean_gradient_plane")
        if not keep_offset:
            result = result - result.mean()
        return result

    mean_value_gradient_x = value_gradient[0].mean()
    mean_value_gradient_y = value_gradient[1].mean()
    for (nx, ny), _ in np.ndenumerate(array2d):
        result[nx, ny] = array2d[nx, ny] - nx * mean_value_gradient_x - ny * mean_value_gradient_y
    if keep_offset:
        result = result + (array2d.mean() - result.mean())
    else:
        result = subtract_mean_level(result)
    return result


class BGCorrectionType(Enum):
    gradient = auto()
    legendre_0 = auto()
    legendre_1 = auto()
    legendre_2 = auto()
    legendre_3 = auto()
    raw_data = auto()


# add all new BGCorrectionType here, so correct background works properly.
# Each function has to accept two parameters, np.ndarray and bool and return np.ndarray.
correct_background_dict = {
    BGCorrectionType.gradient: subtract_mean_gradient_plane,
    BGCorrectionType.legendre_0: partial(subtract_legendre_fit, deg=0),
    BGCorrectionType.legendre_1: partial(subtract_legendre_fit, deg=1),
    BGCorrectionType.legendre_2: partial(subtract_legendre_fit, deg=2),
    BGCorrectionType.legendre_3: partial(subtract_legendre_fit, deg=3),
}


def correct_background(array2d: np.ndarray, correction_type: BGCorrectionType,
                       keep_offset: bool = False) -> Optional[np.ndarray]:
    """
    Returns a numpy.ndarray with corrections given by parameters. Input array2d is not changed.

    :param array2d:
    :param correction_type:
    :param keep_offset:
    :return: ndarray
    """
    if array2d is None:
        return None

    if correction_type == BGCorrectionType.raw_data:
        return array2d.copy()  # return a copy of input data - used in GDEFMeasurement class to restore original values

    return correct_background_dict[correction_type](array2d, keep_offset)


# def average_over_x(array2d: np.ndarray)-> np.ndarray:
#     """
#     Get array with values along y averaged over x.
#     :param array2d:
#     :return:
#     """
