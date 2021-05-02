"""
plotter_utils.py contains functions to create diagrams for AFM measurements.
There are a few naming conventions. Functions starting with 'create_' return a Figure object, while 'plot_ ... _to_ax'
is used for functions expecting an Axes-object to which the data should be plotted. The letter is used e.g. if you
intend to plot several diagrams on a single Figure. Also, by default most functions expect a 2D np.ndarray
for a parameter named data_object, for example 'plot_to_ax'. But they work also when given an Object like
GDEFMeasurement or GDEFSticher. Generally they should work, as long as the given data_object has a values-attribute
which holds the data as a np.ndarray and a pixel_width attribute.
Functions that can plot more than one dataset expect a DataObjectList. That could be the same as DataObject or a List of
DataObject. Additionally it also accepts a dictionary. Here the values have to be of type DataObject and the Keys should
be string - they are used for labeling the corresponding graphs.

For plot_ ... _to_ax, some functions have a default subtitle, which is shown if parameter title='' - to suppress it,
set title=None.

Note that there is a difference between setting a title for plot_ ... _to_ax and create_.... While the first one sets
a subtitle for the Axes, the latter places a Figure suptitle (which might be bigger and placed somewhat different).
@author: Nathanael Jöhrmann
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Union, Literal, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

from gdef_reader.utils import create_xy_rms_data, unit_factor_and_label, get_mu_sigma
from gdef_reporter.plotter_styles import get_plotter_style_rms, PlotterStyle, get_plotter_style_sigma, \
    get_histogram_style_bernhard, get_plotter_style_bernhard, get_plotter_style_histogram

if TYPE_CHECKING:
    from afm_tools.gdef_sticher import GDEFSticher
    from gdef_reader.gdef_measurement import GDEFMeasurement

    DataObject = Union[np.ndarray, GDEFMeasurement, GDEFSticher]  # single data set
    DataDict = dict[str: DataObject]
    DataObjectList = Union[DataObject, DataDict, list[DataObject]]


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


def _get_ax_data_lists(data_object_list: DataObjectList, pixel_width=None, label_list=None,
                       x_units=None) \
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
        pixel_width_list = [pixel_width] * len(data_object_list)
    else:
        pixel_width_list = pixel_width[:]

    if final_label_list is None:
        final_label_list = [None] * len(data_object_list)
    if isinstance(final_label_list, str):
        final_label_list = [final_label_list]

    assert len(final_label_list) == len(data_object_list)
    for i, data in enumerate(data_object_list):
        ndarray2d_data, px_width = _extract_ndarray_and_pixel_width(data, pixel_width_list[i])
        ndarray2d_list.append(ndarray2d_data)
        pixel_width_list[i] = px_width
        if final_label_list[i] is None and hasattr(data, "comment"):
            final_label_list[i] = data.comment

    pixel_width_list, x_units = _check_pixel_width_list(pixel_width_list, x_units)
    return ndarray2d_list, pixel_width_list, final_label_list, x_units


def _check_pixel_width_list(pixel_width_list, x_units):
    if None in pixel_width_list:
        assert all([x is None for x in pixel_width_list]) or x_units == "px", \
            "Cannot mix data with unit [px] with other data"
        x_units = "px"
        pixel_width_list = [1] * len(pixel_width_list)
    return pixel_width_list, x_units


def _copy_plotter_style(plotter_style: PlotterStyle, default: callable) -> PlotterStyle:
    """
    Creates a copy of plotter_style to prevent changing the original data given as parameter.
    If plotter_style is None, creates a new PlotterStyle-object using the callable given to default.
    :param plotter_style: PlotterStyle instance that is copied
    :param default: callable that returns a PlotterStyle - used if plotter_style is None
    :return: PlotterStyle
    """
    if plotter_style is None:
        plotter_style = default()
    else:
        plotter_style = deepcopy(plotter_style)  # do not change input parameter!
    return plotter_style


def best_ratio_fit(total_size: tuple[float, float], single_size: tuple[float, float], n: int):
    """
    Find best ratio of rows and cols to show n axes of ax_size on Figure with total_size.
    :param total_size:
    :param single_size:
    :param n:
    :return:
    """
    optimal_ratio = total_size[0] / total_size[1]

    single_plot_ratio = single_size[0] / single_size[1]
    optimal_ratio /= single_plot_ratio

    possible_ratios = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i * j >= n:
                possible_ratios.append((i, j))
                break

    # sort ratios by best fit to optimal ratio:
    possible_ratios[:] = sorted(possible_ratios, key=lambda ratio: abs(ratio[0] / ratio[1] - optimal_ratio))
    return possible_ratios[0]


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- 2D area plots -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_to_ax(ax: Axes, data_object: DataObject, pixel_width: float = None,
               title: str = "", z_unit: Literal["nm", "µm"] = "nm") -> None:
    """
    Plot values in data_object to given ax.
    :param ax: Axes object to which the surface should be written
    :param data_object: DataObject with surface data
    :param pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)
    :param title: Axes title (if '' -> shows mu and sigma (default); for no title set None)
    :param z_unit: Units for z-Axis (color coded)
    :return: None
    """

    def extent_for_plot(shape, px_width):
        width_in_um = shape[1] * px_width * 1e6
        height_in_um = shape[0] * px_width * 1e6
        return [0, width_in_um, 0, height_in_um]

    z_factor, z_unit_label = unit_factor_and_label(z_unit)
    ndarray2d_data, pixel_width = _extract_ndarray_and_pixel_width(data_object, pixel_width)
    unit = "µm"
    if pixel_width is None:
        pixel_width = 1
        unit = "px"
    extent = extent_for_plot(ndarray2d_data.shape, pixel_width)
    im = ax.imshow(ndarray2d_data * z_factor, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
    ax.set_title(title)  # , pad=16)
    ax.set_xlabel(unit, labelpad=1.0)
    ax.set_ylabel(unit, labelpad=1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_title(z_unit_label, y=1)  # bar.set_label("nm")
    plt.colorbar(im, cax=cax, ax=ax)


def create_plot(data_object: DataObject,
                pixel_width: float = None,
                title: str = '',
                max_figure_size: tuple[float, float] =(4, 4),
                dpi: int = 96,
                cropped: bool = True) \
        -> Figure:
    """
    Creates a matplotlib Figure using given data_object. If cropped is True, the returned Figure has a smaller size
    than specified in max_figure_size.
    :param data_object: DataObject with surface data
    :param pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)
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
                           data_object_list: DataObjectList,
                           pixel_width: Optional[Union[float, list[float]]] = None,
                           label_list: Union[str, list[str]] = None,
                           title: Optional[str] = "",
                           n_bins: int = 200,
                           units: Literal["µm", "nm"] = "µm",
                           add_norm: bool = False,
                           plotter_style=None) \
        -> None:
    """
    Also accepts a list of np.ndarray data (for plotting several histograms stacked)
    :param ax: Axes object to which the surface should be written
    :param data_object_list: DataObject or list[DataObject] with surface data
    :param pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)
    :param label_list: labels for plotted data from values2d
    :param title: Axes title; if empty, mu and sigma will be shown; to prevent any subtitle, set title=None
    :param n_bins: number of equally spaced bins for histogram
    :param units: Can be set to µm or nm (default is µm).
    :param add_norm: if True (default), show normal/gaussian probability density function for each distribution
    :param plotter_style:
    :return: None
    """
    unit_factor, unit_label = unit_factor_and_label(units)
    ndarray2d_list, _, label_list, _ = _get_ax_data_lists(
        data_object_list, pixel_width=pixel_width, label_list=label_list)

    if plotter_style is None:
        plotter_style = get_plotter_style_histogram(nbins=n_bins)
    else:
        plotter_style = deepcopy(plotter_style)
    histogram_styler = plotter_style.hist_styler

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

    for i in range(len(z_values_list)):
        _, _, patch = ax.hist(z_values_list[i], label=label_list[i], **histogram_styler.dict)
        histogram_styler.next_style()

    if add_norm:
        histogram_styler.reset()
        for i, line in enumerate(norm_fit_lines):
            ax.plot(norm_x_values_list[i], line, c=histogram_styler._color())
            histogram_styler.next_style()

    if title == "" and len(ndarray2d_list) == 1:
        title = f"\u03BC={mu:.2f}, \u03C3={sigma:.2f}"

    plotter_style.set(x_unit=unit_label, ax_title=title)
    plotter_style.set_format_to_ax(ax)

    if any(label_list):
        ax.legend()

    return None


def create_z_histogram_plot(data_object_list: DataObjectList,
                            pixel_width=None,
                            labels: Union[str, list[str]] = None,
                            title: Optional[str] = "",
                            n_bins: int = 200,
                            units: Literal["µm", "nm"] = "µm",
                            add_norm: bool = False,
                            plotter_style: PlotterStyle = None) \
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
    :param plotter_style:
    :return: Figure
    """
    if plotter_style is None:
        plotter_style = get_plotter_style_histogram(nbins=n_bins)
    else:
        plotter_style = deepcopy(plotter_style)

    if title:
        plotter_style.set(fig_title=title)
        title = None

    result, ax = plotter_style.create_preformated_figure()

    plot_z_histogram_to_ax(ax, data_object_list, pixel_width, labels, title, n_bins, units, add_norm)
    return result


def plot_rms_to_ax(ax_rms: Axes,
                   data_object_list: DataObjectList,
                   pixel_width=None,
                   label_list: Union[str, list[str]] = None,
                   title: Optional[str] = "",
                   moving_average_n: int = 200,
                   x_offset=0,
                   x_units: Literal["µm", "nm"] = "µm",
                   subtract_average=True,  # <- todo:should this be True or False?
                   plotter_style=None
                   ) \
        -> None:
    """ ... """
    ndarray2d_list, pixel_width_list, label_list, x_units = _get_ax_data_lists(
        data_object_list, pixel_width=pixel_width, label_list=label_list, x_units=x_units)

    _, x_unit_label = unit_factor_and_label(x_units)

    plotter_style = _copy_plotter_style(plotter_style, default=get_plotter_style_rms)
    plotter_style.set(x_unit=x_unit_label, ax_title=title)
    plotter_style.set_format_to_ax(ax_rms)

    graph_styler = plotter_style.graph_styler

    for i, ndarray_data in enumerate(ndarray2d_list):
        x_pos, y_rms = create_xy_rms_data(ndarray_data, pixel_width_list[i], moving_average_n,
                                          subtract_average=subtract_average, x_units=x_units,
                                          y_units=plotter_style.y_unit)

        x_pos = [x + x_offset for x in x_pos]
        ax_rms.plot(x_pos, y_rms, **graph_styler.dict, label=label_list[i])
        graph_styler.next_style()

        if any(label_list):
            ax_rms.legend()


def create_rms_plot(data_object_list: DataObjectList,
                    pixel_width=None,
                    label_list: Union[str, list[str]] = None,
                    title: str = "",
                    moving_average_n: int = 200,
                    x_offset=0,
                    units: Literal["µm", "nm"] = "µm",
                    subtract_average=True,
                    plotter_style: PlotterStyle = None) \
        -> Figure:
    """
    Creates a matplotlib figure, showing a graph of the root mean square of the gradient of the GDEFSticher objects in
    data_dict. The key value in data_dict is used as label in the legend.
    :param data_object_list:
    :param pixel_width: has to be set, if data_object_list contains 1 or more np.ndarry (for varying values, use a list)
    :param label_list:
    :param title:
    :param moving_average_n:
    :param x_offset:
    :param units:
    :param subtract_average:
    :param plotter_style:
    :return:
    """
    plotter_style = _copy_plotter_style(plotter_style, default=get_plotter_style_rms)

    if title:
        info = f"moving average n={moving_average_n}"
        try:
            info = info + f" ({moving_average_n * pixel_width * 1e6:.1f} µm)"
        except TypeError:  # pixel_width is None or list or ...
            pass
        title = f'{title}\n{info}'

    plotter_style.set(fig_title=title)

    result, ax = plotter_style.create_preformated_figure()

    plot_rms_to_ax(ax, data_object_list, pixel_width=pixel_width, label_list=label_list,
                   moving_average_n=moving_average_n, x_offset=x_offset, subtract_average=subtract_average,
                   x_units=units, plotter_style=plotter_style)
    return result


def plot_rms_with_error_to_ax(ax: Axes,
                              data_object_list: DataObjectList,
                              pixel_width=None,
                              label_list: Union[str, list[str]] = None,
                              title: Optional[str] = "",
                              average_n: int = 8,
                              x_units: Literal["px", "µm", "nm"] = "µm",
                              y_units: Literal["µm", "nm"] = "µm",
                              plotter_style: PlotterStyle = None):
    """

    :param ax:
    :param data_object_list:
    :param pixel_width:
    :param label_list:
    :param average_n:
    :param x_units:
    :param y_units:
    :param plotter_style:
    :return:
    """
    ndarray2d_list, pixel_width_list, label_list, x_units = _get_ax_data_lists(
        data_object_list, pixel_width=pixel_width, label_list=label_list, x_units=x_units)

    pixel_width_list, x_units = _check_pixel_width_list(pixel_width_list, x_units)

    x_factor, x_unit_label = unit_factor_and_label(x_units)
    y_factor, y_unit_label = unit_factor_and_label(y_units)

    plotter_style = _copy_plotter_style(plotter_style, default=get_plotter_style_sigma)

    plotter_style.set(x_unit=x_unit_label, y_unit=y_unit_label, ax_title=title)
    graph_styler = plotter_style.graph_styler.reset()
    plotter_style.set_format_to_ax(ax)

    for i, data in enumerate(ndarray2d_list):
        z_data = data

        # get mu for every column first:
        sigma_col_list = []
        for j in range(0, z_data.shape[1]):
            _, sigma_col = get_mu_sigma(z_data[:, j:j + 1])
            sigma_col_list.append(sigma_col)

        x_pos = []
        y_rms = []
        y_error = []
        pixel_width = pixel_width_list[i] * x_factor
        for j in range(0, z_data.shape[1] - average_n, average_n):  # step):
            x_pos.append((j + max(average_n - 1, 0) / 2.0) * pixel_width)

            mu_rms, sigma_rms = get_mu_sigma(np.array(sigma_col_list[j:j + average_n]))
            y_rms.append(mu_rms * y_factor)
            y_error.append(sigma_rms * y_factor)
        style_dict = {
            "fmt": 'o',
            "elinewidth": 0.6,
            "capsize": 2.0,
            "markersize": 5,
            "color": graph_styler.dict["color"]
        }
        ax.errorbar(x_pos, y_rms, yerr=y_error, label=label_list[i],
                    **style_dict)  # **graph_styler.dict, label=key)  #fmt='-o')  # **graph_styler.dict
        graph_styler.next_style()
    # ax_rms.set_title(f"window width = {moving_average_n*pixel_width_in_um:.1f}")
    if any(label for label in label_list if label is None):
        ax.legend()
    ax.legend()


def create_rms_with_error_plot(data_object_list: DataObjectList,
                               pixel_width=None,
                               label_list: Union[str, list[str]] = None,
                               title: Optional[str] = "",
                               average_n: int = 8,
                               x_units: Literal["px", "µm", "nm"] = "µm",
                               y_units: Literal["µm", "nm"] = "µm",
                               plotter_style: PlotterStyle = None) \
        -> Figure:
    """

    :param data_object_list:
    :param pixel_width:
    :param label_list:
    :param title:
    :param average_n:
    :param x_units:
    :param y_units:
    :param plotter_style:
    :return:
    """
    plotter_style = _copy_plotter_style(plotter_style, default=get_plotter_style_sigma)

    plotter_style.set(fig_title=title)
    result, ax_rms = plotter_style.create_preformated_figure()
    plot_rms_with_error_to_ax(ax_rms, data_object_list, pixel_width=pixel_width, label_list=label_list,
                              average_n=average_n, x_units=x_units, y_units=y_units, plotter_style=plotter_style)
    result.tight_layout()
    return result


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- misc -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_summary_plot(data_object_list: DataObjectList,
                        pixel_width: Optional[float] = None,
                        ax_title_list: Union[str, list[str]] = None,
                        title: Optional[str] = "",
                        figure_size: tuple[float, float] = (16, 10),
                        dpi: int = 96):
    """
    Creates a Figure with stiched maps for each GDEFSticher in sticher_dict. The keys in sticher_dict
    are used as titles for the corresponding Axes.

    :param data_object_list:
    :param pixel_width:
    :param ax_title_list:
    :param title:
    :param figure_size:
    :param dpi:
    :return:
    """
    ndarray2d_list, pixel_width_list, ax_title_list, _ = _get_ax_data_lists(data_object_list,
                                                                            pixel_width=pixel_width,
                                                                            label_list=ax_title_list)
    n = len(ndarray2d_list)
    if n == 0:
        result, _ = plt.subplots(1, figsize=figure_size, dpi=dpi)
        return result

    # dummy_fig is only needed to estimate aspect ratio of a single axe
    dummy_fig = create_plot(ndarray2d_list[0], pixel_width_list[0], title='dummy',
                            max_figure_size=figure_size, cropped=True)

    n_cols, n_rows = best_ratio_fit(figure_size, dummy_fig.get_size_inches(), n)
    result, ax_list = plt.subplots(n_rows, n_cols, figsize=figure_size, dpi=dpi)

    if not isinstance(ax_list, np.ndarray):
        ax_list = np.asarray([ax_list])

    for i, data_object in enumerate(ndarray2d_list):
        plot_to_ax(ax_list.flatten('F')[i], data_object, pixel_width_list[i], title=ax_title_list[i])

    for ax in ax_list.flatten('F')[n:]:
        ax.set_axis_off()

    if title:
        result.suptitle(title)

    result.tight_layout()
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
def _get_greyscale_data(data_object: DataObject, alpha=0):
    ndarray2d_data, _ = _extract_ndarray_and_pixel_width(data_object)
    # Normalised [0,1]
    data_min = np.min(ndarray2d_data)
    data_ptp = np.ptp(ndarray2d_data)

    result = np.zeros((ndarray2d_data.shape[0], ndarray2d_data.shape[1], 4))
    for (nx, ny), _ in np.ndenumerate(ndarray2d_data):
        value = (ndarray2d_data[nx, ny] - data_min) / data_ptp
        result[nx, ny] = (value, value, value, alpha)
    return result


# todo: used/intended for what?
def _create_image_data(data_object: DataObject):
    ndarray2d_data, _ = _extract_ndarray_and_pixel_width(data_object)
    data_min = np.nanmin(ndarray2d_data)
    # normalize the data to 0 - 1:
    array2d = (ndarray2d_data - min(0, data_min)) / (np.nanmax(ndarray2d_data) - min(0, data_min))
    array2d = 255 * array2d  # Now scale by 255
    return array2d.astype(np.uint8)

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
