import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gdef_reader.utils import create_absolute_gradient_array, create_xy_rms_data


def plot_surface_to_axes(ax: Axes, values: np.ndarray, pixel_width: float,
                         title="", z_unit="nm", z_factor=1e9) -> None:
    """
    Plot surface-values to given ax. Necessary, to use figures with subplots effectivly.

    :param ax: Axes object to which the surface should be written
    :param values: np.ndarray (2D array) with surface data
    :param pixel_width: Pixel width/height in [m]
    :param title: Axes title
    :param z_unit: Units for z-Axis (color coded)
    :param z_factor: scaling factor for z-values (e.g. 1e9 for m -> nm)
    :return: None
    """

    def extent_for_plot(shape, pixel_width):
        width_in_um = shape[1] * pixel_width * 1e6
        height_in_um = shape[0] * pixel_width * 1e6
        return [0, width_in_um, 0, height_in_um]

    extent = extent_for_plot(values.shape, pixel_width)
    im = ax.imshow(values * z_factor, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
    ax.set_title(title)  # , pad=16)
    ax.set_xlabel("µm", labelpad=1.0)
    ax.set_ylabel("µm", labelpad=1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_title(z_unit, y=1)  # bar.set_label("nm")
    plt.colorbar(im, cax=cax)


# ----------------------------------------------
# todo: below here functions still need clean up
# ----------------------------------------------

def get_compare_gradient_rms_figure(cls, sticher_dict, cutoff_percent=8, moving_average_n=1, figsize=(8, 4),
                                    x_offset=0):
    fig, ax_compare_gradient_rms = plt.subplots(1, figsize=figsize, dpi=300)

    ax_compare_gradient_rms.set_xlabel("[µm]")
    ax_compare_gradient_rms.set_ylabel(
        f"roughness(gradient) (moving average n = {moving_average_n})")
    ax_compare_gradient_rms.set_yticks([])
    counter = 0
    for key in sticher_dict:
        sticher = sticher_dict[key]

        absolute_gradient_array = create_absolute_gradient_array(sticher.stiched_data, cutoff_percent / 100.0)
        x_pos, y_gradient_rms = create_xy_rms_data(absolute_gradient_array, sticher.pixel_width,
                                                   moving_average_n)
        x_pos = [x + x_offset for x in x_pos]
        ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=key)

        # if counter == 0:
        #     ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=f"fatigued", color="black")  # f"{cutoff_percent}%")
        #     counter = 1
        # else:
        #     ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=f"pristine", color="red")  # f"{cutoff_percent}%")

        ax_compare_gradient_rms.legend()
    # fig.suptitle(f"cutoff = {cutoff_percent}%")
    fig.tight_layout()

