"""
plotter_utils.py contains functions to create diagrams for AFM measurements.
There are a few naming conventions. Functions starting with 'create_' return a Figure object, while 'plot_ ... _to_ax'
is used for functions expecting an Axes-object to which the data should be plotted. The letter is used e.g. if you
intend to plot several diagrams on a single Figure. Also, by default most functions expect a 2D np.ndarray
for a parameter named values2d, for example 'plot_to_ax'. But they work also when given a GDEFMeasurement or
GDEFSticher. All functions expecting a different data type have to state this in the function name
by adding '_from_...', e.g. ...

For plot_ ... _to_ax, some functions have a default subtitle, which is shown if parameter title='' - to suppress it,
set title=None.

Note that there is a difference between setting a title for plot_ ... _to_ax and create_.... While the first one sets
a subtitle for the Axes, the latter places a Figure suptitle (which might be bigger and placed somewhat different).
@author: Nathanael Jöhrmann
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union, Literal, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

from gdef_reader.utils import create_xy_rms_data

if TYPE_CHECKING:
    from afm_tools.gdef_sticher import GDEFSticher
    from gdef_reader.gdef_measurement import GDEFMeasurement

    DataObject = Union[np.ndarray, GDEFMeasurement, GDEFSticher]  # single data set
    DataDict = dict[str: DataObject]
    DataObjectList = Union[DataObject, DataDict, list[DataObject]]


def _unit_factor_and_label(units: Literal["µm", "nm"]) -> tuple[float, str]:
    units_dict = {
        "nm": (1e9, "nm"),
        "µm": (1e6, "\u03BCm")
    }
    _units = units.replace("\u03BC", "µ")  # \u03BC is not equal to µ!
    return units_dict[_units]


def _get_tight_size(max_figure: Figure, title: str):
    """get cropped size for max_figure after adding optional title"""
    # first, place suptitle close to axes
    if title:
        tight_bbox = max_figure.get_tightbbox(max_figure.canvas.get_renderer())
        y_rel = tight_bbox.ymax / max_figure.bbox_inches.ymax
        max_figure.suptitle(title, y=y_rel + 0.02, verticalalignment="bottom")

    max_figure.tight_layout()
    tight_bbox = max_figure.get_tightbbox(max_figure.canvas.get_renderer())
    max_figure_size = max_figure.bbox_inches.size

    # only crop in one direction
    if tight_bbox.size[0] / max_figure_size[0] < tight_bbox.size[1] / max_figure_size[1]:
        return tight_bbox.size[0], max_figure_size[1]
    else:
        return max_figure_size[0], tight_bbox.size[1]


def _add_suptitle_and_tighten(figure, title) -> Figure:
    """
    Add given title as suptitle to Figure, and finally calls tight_layout() on it.
    :param figure:
    :param title:
    :return:
    """
    if title:
        figure.suptitle(title)
    figure.tight_layout(pad=0.5)
    return figure


def _extract_ndarray_and_pixel_width(data_object: DataObject,
                                     pixel_width=None) -> (np.ndarray, float):
    """
    Tries to extract np.ndarrray and pixel_width from given values_object, by looking for attributes 'values' and
    'pixel_width'.
    :param data_object: np.ndarray or object with attribute values (and pixel_width, if parameter pixel_width is None)
    :param pixel_width: pixel_width or None (is ignored, if values_object is not a np.ndarray)
    :return: Tuple containing np.ndarray and pixel_width
    """
    if isinstance(data_object, np.ndarray):
        return data_object, pixel_width

    return data_object.values, data_object.pixel_width


def _get_ndarray_pixel_width_and_label_lists(data_object_list: DataObjectList, pixel_width=None, label_list=None) \
        -> (list[np.ndarray], list[float], list[str]):
    """
    Tries to extract a np.ndarray list and a pixel_width list from given values_object_list,
    by looking for attributes 'values' and 'pixel_width'. Used when plotting more than one dataset.
    If data_object_list is a dict, first extracts a key_list, used as label_list, if label_list is None.
    """
    final_label_list = label_list
    if isinstance(data_object_list, dict):
        data_object_list, key_list = split_dict_in_data_and_label_list(data_object_list)
        if label_list is None:
            final_label_list = key_list

    ndarray2d_list = []
    pixel_width_list = []
    if not isinstance(data_object_list, list):
        data_object_list = [data_object_list]
    if not isinstance(pixel_width, list):
        pixel_width = [pixel_width] * len(data_object_list)
    for i, data in enumerate(data_object_list):
        ndarray2d_data, px_width = _extract_ndarray_and_pixel_width(data, pixel_width)
        ndarray2d_list.append(ndarray2d_data)
        pixel_width_list.append(px_width)

    if final_label_list is None:
        final_label_list = [None] * len(ndarray2d_list)

    if isinstance(final_label_list, str):
        final_label_list = [final_label_list]

    assert len(final_label_list) == len(ndarray2d_list)

    return ndarray2d_list, pixel_width_list, final_label_list


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- 2D area plots -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_to_ax(ax: Axes, data_object: DataObject, pixel_width: float = None,
               title: str = "", z_unit: Literal["nm", "µm"] = "nm") -> None:
    """
    Plot values in values2d to given ax.
    :param ax: Axes object to which the surface should be written
    :param data_object: DataObject with surface data
    :param pixel_width: Pixel width/height in [m]
    :param title: Axes title (if '' -> shows mu and sigma (default); for no title set None)
    :param z_unit: Units for z-Axis (color coded)
    :return: None
    """

    def extent_for_plot(shape, px_width):
        width_in_um = shape[1] * px_width * 1e6
        height_in_um = shape[0] * px_width * 1e6
        return [0, width_in_um, 0, height_in_um]

    z_factor, z_unit_label = _unit_factor_and_label(z_unit)
    ndarray2d_data, pixel_width = _extract_ndarray_and_pixel_width(data_object, pixel_width)

    extent = extent_for_plot(ndarray2d_data.shape, pixel_width)
    im = ax.imshow(ndarray2d_data * z_factor, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
    ax.set_title(title)  # , pad=16)
    ax.set_xlabel("µm", labelpad=1.0)
    ax.set_ylabel("µm", labelpad=1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_title(z_unit_label, y=1)  # bar.set_label("nm")
    plt.colorbar(im, cax=cax, ax=ax)


def create_plot(data_object: DataObject, pixel_width: float = None, title: str = '',
                max_figure_size=(4, 4), dpi=96, cropped=True) -> Figure:
    """
    Creates a matplotlib Figure using given values2d-object. If cropped is True, the returned Figure has a smaller size
    than specified in max_figure_size.
    :param data_object: DataObject with surface data
    :param pixel_width: Pixel width/height in [m]
    :param title: optional title (implemented as Figure suptitle)
    :param max_figure_size: Max. figure size of returned Figure (actual size might be smaller if cropped).
    :param dpi: dpi value of returned Figure
    :param cropped: Crop the result Figure (default is True). Useful if aspect ratio of Figure and plot differ.
    :return: Figure
    """
    figure_max, ax = plt.subplots(figsize=max_figure_size, dpi=dpi)
    plot_to_ax(ax=ax, data_object=data_object, pixel_width=pixel_width)  # , title=title)

    if not cropped:  # only add suptitle if not cropped, otherwise _get_tight_size() cannot get correct cropped size!
        return _add_suptitle_and_tighten(figure_max, title)

    cropped_size = _get_tight_size(figure_max, title)
    return create_plot(data_object, pixel_width, title, max_figure_size=cropped_size, dpi=dpi, cropped=False)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- 1D plots over x ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_z_histogram_to_ax(ax: Axes,
                           data_object_list: DataObject,
                           label_list: Union[str, list[str]] = None,
                           title: Optional[str] = "",
                           n_bins: int = 200,
                           units: Literal["µm", "nm"] = "µm",
                           add_norm: bool = False)\
        -> None:
    """
    Also accepts a list of np.ndarray data (for plotting several histograms stacked)
    :param ax: Axes object to which the surface should be written
    :param data_object_list: DataObject or list[DataObject] with surface data
    :param label_list: labels for plotted data from values2d
    :param title: Axes title; if empty, mu and sigma will be shown; to prevent any subtitle, set title=None
    :param n_bins: number of equally spaced bins for histogram
    :param units: Can be set to µm or nm (default is µm).
    :param add_norm: if True (default), show normal/gaussian probability density function for each distribution
    :return: None
    """
    unit_factor, unit_label = _unit_factor_and_label(units)
    ndarray2d_list, _, label_list = _get_ndarray_pixel_width_and_label_lists(data_object_list, label_list)

    colors = []
    z_values_list = []
    norm_fit_lines = []
    norm_x_values_list = []
    for ndarray_data in ndarray2d_list:
        z_values = ndarray_data.flatten()
        z_values = z_values[
            ~np.isnan(z_values)]  # remove all NaN values (~ is bitwise NOT operator - similar to numpy.logical_not)
        z_values = z_values * unit_factor  # m -> µm/nm
        mu, sigma = norm.fit(z_values)
        norm_x_values = np.linspace(z_values.min(), z_values.max(), 100)
        norm_fit_line = norm.pdf(norm_x_values, mu, sigma)
        z_values_list.append(z_values)
        norm_fit_lines.append(norm_fit_line)
        norm_x_values_list.append(norm_x_values)
        # todo: implement proper solution for colors
        if len(colors) % 2 > 0:
            colors.append("red")
        else:
            colors.append("black")

    for i in range(len(z_values_list)):
        _, _, patch = ax.hist(z_values_list[i], density=True, bins=n_bins, edgecolor=colors[i], lw=1,
                              label=label_list[i],
                              fc=to_rgba(colors[i], alpha=0.3), rwidth=1, histtype="bar", fill=True)

    if add_norm:
        for i, line in enumerate(norm_fit_lines):
            ax.plot(norm_x_values_list[i], line, c=colors[i])

    # todo: consider a PlotterStyle instead
    ax.set_xlabel(f'z [{unit_label}]')
    ax.set_ylabel('Normalized counts')
    # ax.grid(True)
    if title:
        ax.set_title(f"{title}")
    elif title is not None and len(ndarray2d_list) == 1:
        ax.set_title(f"\u03BC={mu:.2f}, \u03C3={sigma:.2f}")

    if any(label_list):
        ax.legend()

    return None


def create_z_histogram_plot(data_object_list: DataObjectList,
                            labels: Union[str, list[str]] = None,
                            title: Optional[str] = "",
                            n_bins: int = 200,
                            units: Literal["µm", "nm"] = "µm",
                            add_norm: bool = False,
                            figure_size: tuple[float, float] = (6, 3),
                            dpi: int = 96) \
        -> Figure:
    """
    Also accepts a list of np.ndarray data (for plotting several histograms stacked)
    :param data_object_list:DataObjectList
    :param labels: labels for plotted data from values2d
    :param title: Figure title; if empty, mu and sigma will be shown as axes subtitle(use title=None to prevent this)
    :param n_bins: number of equally spaced bins for histogram
    :param units: Can be set to µm or nm (default is µm).
    :param add_norm: if True (default), show normal/gaussian probability density function for each distribution
    :param figure_size:
    :param dpi:
    :return: Figure
    """
    result, ax = plt.subplots(1, 1, figsize=figure_size, tight_layout=True, dpi=dpi)
    if title:
        result.suptitle(title)
        title = None

    plot_z_histogram_to_ax(ax, data_object_list, labels, title, n_bins, units, add_norm)
    return result


def plot_rms_to_ax(ax_rms: Axes,
                   data_object_list: DataObjectList,
                   label_list: Union[str, list[str]] = None,
                   title: Optional[str] = "",
                   moving_average_n: int = 200,
                   x_offset=0,
                   units: Literal["µm", "nm"] = "µm",
                   subtract_average=True  # <- todo:should this be True or False?
                   )\
        -> None:
    # graph_styler = plotter_style.graph_styler

    ax_rms.set_xlabel("[µm]")
    y_label = f"roughness (moving average n = {moving_average_n})"
    ax_rms.set_ylabel(y_label)
    ax_rms.set_yticks([])
    ax_rms.set_title(title)

    ndarray2d_list, pixel_width_list, label_list = _get_ndarray_pixel_width_and_label_lists(data_object_list,
                                                                                            label_list=label_list)
    for i, ndarray_data in enumerate(ndarray2d_list):
        x_pos, y_rms = create_xy_rms_data(ndarray_data, pixel_width_list[i], moving_average_n,
                                          subtract_average=subtract_average)

        x_pos = [x + x_offset for x in x_pos]
        ax_rms.plot(x_pos, y_rms, label=label_list[i])
        # ax_rms.plot(x_pos, y_rms, **graph_styler.dict, label=key)
        # graph_styler.next_style()

        if any(label_list):
            ax_rms.legend()


def create_rms_plot(data_object_list: DataObjectList,
                    labels: Union[str, list[str]] = None,
                    title: Optional[str] = "",
                    moving_average_n: int = 200,
                    x_offset=0,
                    units: Literal["µm", "nm"] = "µm",
                    subtract_average=True,
                    figure_size: tuple[float, float] = (6, 3),
                    dpi: int = 96)\
        -> Figure:
    """
    Creates a matplotlib figure, showing a graph of the root meean square of the gradient of the GDEFSticher objects in
    data_dict. The key value in data_dict is used as label in the legend.
    :param data_object_list:
    :param labels:
    :param title:
    :param moving_average_n:
    :param x_offset:
    :param subtract_average:
    :param figure_size:
    :param dpi:
    :return:
    """
    result, ax = plt.subplots(1, 1, figsize=figure_size, tight_layout=True, dpi=dpi)
    if title:
        result.suptitle(title)
        title = None

    plot_rms_to_ax(ax, data_object_list, title=title, label_list=labels, moving_average_n=moving_average_n,
                   x_offset=x_offset, subtract_average=subtract_average)
    return result


def split_dict_in_data_and_label_list(data_dict_list: dict[str: DataObject]):
    label_list = []
    data_object_list = []
    for key, value in data_dict_list.items():
        label_list.append(key)
        data_object_list.append(value)
    return data_object_list, label_list


def save_figure(figure: Figure, output_path: Path, filename: str, png: bool = True, pdf: bool = False) -> None:
    """
    Helper function to save a matplotlib figure as png and or pdf. Automatically creates output_path, if necessary.
    Does nothing if given output_path is None.
    """
    if not output_path:
        return None
    if pdf or png:
        output_path.mkdir(parents=True, exist_ok=True)
    if png:
        figure.savefig(output_path.joinpath(f"{filename}.png"))  # , dpi=300)
    if pdf:
        figure.savefig(output_path.joinpath(f"{filename}.pdf"))


# -------------------------------------------------------------------
# todo: below here functions still need clean up  or might be removed
# -------------------------------------------------------------------

# todo: used/intended for what?
def _get_greyscale_data(values2d: np.ndarray, alpha=0):
    # Normalised [0,1]
    data_min = np.min(values2d)
    data_ptp = np.ptp(values2d)

    result = np.zeros((values2d.shape[0], values2d.shape[1], 4))
    for (nx, ny), _ in np.ndenumerate(values2d):
        value = (values2d[nx, ny] - data_min) / data_ptp
        result[nx, ny] = (value, value, value, alpha)
    return result

# def get_compare_gradient_rms_figure(cls, sticher_dict, cutoff_percent=8, moving_average_n=1, figsize=(8, 4),
#                                     x_offset=0):
#     fig, ax_compare_gradient_rms = plt.subplots(1, figsize=figsize, dpi=300)
#
#     ax_compare_gradient_rms.set_xlabel("[µm]")
#     ax_compare_gradient_rms.set_ylabel(
#         f"roughness(gradient) (moving average n = {moving_average_n})")
#     ax_compare_gradient_rms.set_yticks([])
#     counter = 0
#     for key in sticher_dict:
#         sticher = sticher_dict[key]
#
#         absolute_gradient_array = create_absolute_gradient_array(sticher.stiched_data, cutoff_percent / 100.0)
#         x_pos, y_gradient_rms = create_xy_rms_data(absolute_gradient_array, sticher.pixel_width,
#                                                    moving_average_n)
#         x_pos = [x + x_offset for x in x_pos]
#         ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=key)
#
#         # if counter == 0:
#         #     ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=f"fatigued", color="black")  # f"{cutoff_percent}%")
#         #     counter = 1
#         # else:
#         #     ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, label=f"pristine", color="red")  # f"{cutoff_percent}%")
#
#         ax_compare_gradient_rms.legend()
#     # fig.suptitle(f"cutoff = {cutoff_percent}%")
#     fig.tight_layout()

#
#
#
#
#
# def create_gradient_rms_figure(sticher_dict: Dict[str, GDEFSticher], cutoff_percent=8, moving_average_n=1,
#                                x_offset=0, plotter_style: PlotterStyle = None) -> Figure:
#     """
#     Creates a matplotlib figure, showing a graph of the root meean square of the gradient of the GDEFSticher objects in
#     data_dict. The key value in data_dict is used as label in the legend.
#     :param sticher_dict:
#     :param cutoff_percent:
#     :param moving_average_n:
#     :param x_offset:
#     :param plotter_style:
#     :return:
#     """
#     if plotter_style is None:
#         plotter_style = PlotterStyle(300, (8, 4))
#     y_label = f"roughness(gradient) (moving average n = {moving_average_n})"
#
#     data_dict = {}
#     for key, sticher in sticher_dict.items():
#         gradient_data = create_absolute_gradient_array(sticher.stiched_data, cutoff_percent / 100.0)
#         data_dict[key] = {"pixel_width": sticher.pixel_width, "data": gradient_data}
#     result = _create_rms_figure(data_dict, moving_average_n, x_offset, plotter_style, y_label)
#     return result
#
#
#
#
#
# # def create_image_data(array2d):
# #     data_min = np.nanmin(array2d)
# #     array2d = (array2d - min(0, data_min)) / (np.nanmax(array2d) - min(0, data_min))  # normalize the data to 0 - 1
# #     array2d = 255 * array2d  # Now scale by 255
# #     return array2d.astype(np.uint8)
