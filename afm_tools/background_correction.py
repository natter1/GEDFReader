import numpy as np
from numpy.polynomial import Legendre
import matplotlib.pyplot as plt

def main():
    array2d = np.array([[1, 4, 1.67], [2, 3, 2]])
    subtract_legendre_fit(array2d, 0)


def correct_background(legendre_deg: int = 0, median_level: bool = True) -> np.ndarray:
    """
    Returns a numpy.ndarray with corrections given by parameters.
    :param median_level: If True, average value is subtracted from all data points.
    :param legendre_deg:
    """
    pass#_do_median_level(subtract_mean_plane=True)
    # background_corrected = True


# def average_over_x(array2d: np.ndarray)-> np.ndarray:
#     """
#     Get array with values along y averaged over x.
#     :param array2d:
#     :return:
#     """


def subtract_mean_level(array2d: np.ndarray):
    result =array2d
    try:
        result = array2d - array2d.mean()
    except ValueError:
        print(f"ValueError in subtract_median_level")
        pass
    return result


def subtract_legendre_fit(array2d: np.ndarray, deg: int = 1, keep_offset: bool = False) -> np.ndarray:
    """
    Use a legendre polynomial fit of degree legendre_deg in X and Y direction to correct background.
    legendre_deg = 0 ... subtract mean value
    legendre_deg = 1 ... subtract mean plane
    legendre_deg = 2 ... subtract simple curved mean surface
    legendre_deg = 3 ... also corrects "s-shaped" distortion
    ...
    """
    result = array2d
    if deg == 0 and keep_offset:
        return result
    n_row = np.linspace(-1, 1, array2d.shape[0])
    n_col = np.linspace(-1, 1, array2d.shape[1])
    mean_row = array2d.mean(axis=1)
    mean_col = array2d.mean(axis=0)

    fit_x = Legendre.fit(n_row, mean_row, deg)
    fit_y = Legendre.fit(n_col, mean_col, deg)

    result = (result.transpose() - np.polynomial.legendre.legval(n_row, fit_x.coef)).transpose()
    result = result - np.polynomial.legendre.legval(n_col, fit_y.coef)
    if keep_offset:
        result = result + 2*array2d.mean()  # mean was subtracted 2 times (once for fit_x ans once for fit_y)
    else:
        result = result + array2d.mean()
    return result


def subtract_mean_gradient_plane(array2d: np.ndarray, keep_offset: bool = False) -> np.ndarray:
    """
    Returns 2d numpy.ndarray with subtracted mean gradient plane from given array2d. Using the gradient might give
     better results, when the measurement has asymmetric structures like large objects on a surface.
                                  ____________________
    example: ____________________|                   |__
    """
    result = array2d
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
        result = result + (array2d.mean() + result.mean())
    else:
        result = subtract_mean_level(result)
    return result


if __name__ == "__main__":
    main()