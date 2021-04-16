import copy
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.stats import norm
from matplotlib.figure import Figure
# todo: optional import:
from pptx_tools.creator import PPTXCreator
from pptx_tools.templates import AbstractTemplate

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_indent_analyzer import GDEFIndentAnalyzer
from gdef_reader.gdef_measurement import GDEFMeasurement
from gdef_reader.gdef_sticher import GDEFSticher
from gdef_reader.plotter_styles import PlotterStyle
from gdef_reader.pptx_styles import summary_table, position_2x2_00, position_2x2_10, position_2x2_01, \
    minimize_table_height, position_2x2_11


def create_pygdf_files(input_path: Path, output_path: Path = None, create_images: bool = False) -> List[Path]:
    result = []
    gdf_filenames = input_path.glob("*.gdf")  # glob returns a generator, so gdf_filenames can only be used once!

    if not output_path:
        output_path = input_path.joinpath("pygdf")
    output_path.mkdir(parents=True, exist_ok=True)

    for gdf_filename in gdf_filenames:
        gdf_importer = GDEFImporter(gdf_filename)
        pygdf_path = output_path.joinpath(gdf_filename.stem)
        result.append(pygdf_path)
        gdf_importer.export_measurements(pygdf_path, create_images)

    return result


def load_pygdf_measurements(path: Path) -> List[GDEFMeasurement]:
    result = []
    # files = path.rglob("*.pygdf")  # includes subfolders
    files = path.glob("*.pygdf")
    for filename in files:
        print(filename)
        with open(filename, 'rb') as file:
            measurement = pickle.load(file)
            measurement.pygdf_filename = filename
            result.append(measurement)
    return result


def create_png_for_nanoindents(path: Path, png_save_path: Optional[Path] = None):
    measurements = load_pygdf_measurements(path)
    if png_save_path is None:
        png_save_path = path
    else:
        png_save_path = png_save_path
        png_save_path.mkdir(exist_ok=True)
    for measurement in measurements:
        indent_analyzer = GDEFIndentAnalyzer(measurement)
        print(measurement.comment)
        figure = measurement.create_plot()
        if figure is None:
            continue
        print(png_save_path.joinpath(f"{measurement.pygdf_filename.stem + '.png'}"))
        figure.savefig(png_save_path.joinpath(f"{measurement.pygdf_filename.stem + '.png'}"),
                       dpi=96)  # , transparent=transparent)
        indent_analyzer.add_indent_pile_up_mask_to_axes(figure.axes[0])
        print(png_save_path.joinpath(f"{measurement.pygdf_filename.stem + '_masked.png'}"))
        figure.savefig(png_save_path.joinpath(f"{measurement.pygdf_filename.stem + '_masked.png'}"), dpi=96)
        figure.clear()


def create_pptx_for_nanoindents(path, pptx_filename, pptx_template: Optional[AbstractTemplate] = None):
    pptx = PPTXCreator(template=pptx_template)
    pptx.add_title_slide(f"AFM on Nanoindents - {path.stem}")
    measurements = load_pygdf_measurements(path)
    for measurement in measurements:
        indent_analyzer = GDEFIndentAnalyzer(measurement)
        print(measurement.comment)
        slide = pptx.add_slide(measurement.comment)

        figure = measurement.create_plot()
        if figure is None:
            continue
        pptx.add_matplotlib_figure(figure, slide, position_2x2_00())
        table_shape = pptx.add_table(slide, measurement.get_summary_table_data(), position_2x2_01(),
                                     table_style=summary_table())
        minimize_table_height(table_shape)
        # figure.savefig(f"{measurement.basename.with_suffix('.png')}")  # , transparent=transparent)

        indent_analyzer.add_indent_pile_up_mask_to_axes(figure.axes[0], roughness_part=0.05)
        # figure.savefig(f"{measurement.basename.with_name(measurement.basename.stem + '_masked.png')}", dpi=96)
        pptx.add_matplotlib_figure(figure, slide, position_2x2_10())
        table_shape = pptx.add_table(slide, indent_analyzer.get_summary_table_data(), position_2x2_11(),
                                     table_style=summary_table())
        minimize_table_height(table_shape)
        figure.clear()
    pptx.save(path.joinpath(f"{pptx_filename}.pptx"),
              overwrite=True)  # todo: remove overwrite=True when testing is finished


# get rms roughness
def nanrms(x: np.ndarray, axis=None, subtract_average: bool = False):
    """Returns root mean square of given numpy.ndarray x."""
    average = 0
    if subtract_average:
        average = np.nanmean(x)  # don't do this with rms for gradient field!
    return np.sqrt(np.nanmean((x - average) ** 2, axis=axis))


def create_absolute_gradient_array(array2d, cutoff=1.0):
    result = np.gradient(array2d)
    result = np.sqrt(result[0] ** 2 + result[1] ** 2)
    max_grad = np.nanmax(result)
    with np.nditer(result, op_flags=['readwrite']) as it:
        for x in it:
            if not np.isnan(x) and x > cutoff * max_grad:
                x[...] = np.nan
    return result


# def create_image_data(array2d):
#     data_min = np.nanmin(array2d)
#     array2d = (array2d - min(0, data_min)) / (np.nanmax(array2d) - min(0, data_min))  # normalize the data to 0 - 1
#     array2d = 255 * array2d  # Now scale by 255
#     return array2d.astype(np.uint8)

def create_xy_rms_data(values: np.ndarray, pixel_width, moving_average_n=1, subtract_average=False) -> Tuple[list, list]:
    """
    :param values: 2D array
    :param pixel_width:
    :param moving_average_n:
    :return: (x_pos, y_rms)
    """
    x_pos = []
    y_rms = []
    for i in range(values.shape[1] - moving_average_n):
        x_pos.append((i + max(moving_average_n - 1, 0) / 2.0) * pixel_width * 1e6)
        y_rms.append(nanrms(values[:, i:i + moving_average_n], subtract_average=subtract_average))
    return x_pos, y_rms


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
        figure.savefig(output_path.joinpath(f"{filename}.png"), dpi=300)
    if pdf:
        figure.savefig(output_path.joinpath(f"{filename}.pdf"))


# def _create_figure(data_dict: Dict[str, dict], plotter_style: PlotterStyle, x_label: str, y_label: str,
#                    title: str = None) -> Figure:
#     """
#
#     :param data_dict:
#     :param plotter_style:
#     :param x_label:
#     :param y_label:
#     :param title:
#     :return:
#     """
#     graph_styler = plotter_style.graph_styler
#
#     result, ax = plt.subplots(1, figsize=plotter_style.figure_size, dpi=plotter_style.dpi)
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     if title:
#         result.suptitle(title)
#     # ax.set_yticks([])
#     for key, value in data_dict.items():
#         x_pos, y_rms = create_xy_rms_data(value["data"], value["pixel_width"], moving_average_n)
#         x_pos = [x + x_offset for x in x_pos]
#
#         ax.plot(x_pos, y_rms, **graph_styler.dict, label=key)
#         graph_styler.next_style()
#
#         ax.legend()
#
#     # fig.suptitle(f"cutoff = {cutoff_percent}%")
#     result.tight_layout()
#     return result


def _create_rms_figure(data_dict: Dict[str, dict], moving_average_n, x_offset,
                       plotter_style: PlotterStyle, y_label: str, subtract_average=False) -> Figure:
    """
    Creates a matplotlib figure, showing a graph of the root meean square of the np.ndarray in
    data_dict. The key value in data_dict is used as label in the legend.
    :param data_dict: key: label for legend entry; value: dict with entries for pixel_width and data
    :param moving_average_n: n is the number of cols used for moving average.
    :param x_offset: moing graphs along x-axis if neccessary (e.g. to align different measurements)
    :param plotter_style:
    :return: Figure
    """
    graph_styler = plotter_style.graph_styler

    result, ax_rms = plt.subplots(1, figsize=plotter_style.figure_size, dpi=plotter_style.dpi)

    ax_rms.set_xlabel("[µm]")
    ax_rms.set_ylabel(y_label)
    ax_rms.set_yticks([])

    for key, value in data_dict.items():
        x_pos, y_rms = create_xy_rms_data(value["data"], value["pixel_width"], moving_average_n, subtract_average=subtract_average)
        x_pos = [x + x_offset for x in x_pos]

        ax_rms.plot(x_pos, y_rms, **graph_styler.dict, label=key)
        graph_styler.next_style()

        ax_rms.legend()

    # fig.suptitle(f"cutoff = {cutoff_percent}%")
    result.tight_layout()
    return result


def create_rms_figure(sticher_dict: Dict[str, GDEFSticher], moving_average_n=1,
                      x_offset=0, plotter_style: PlotterStyle = None) -> Figure:
    """
    Creates a matplotlib figure, showing a graph of the root meean square of the gradient of the GDEFSticher objects in
    data_dict. The key value in data_dict is used as label in the legend.
    :param sticher_dict:
    :param cutoff_percent:
    :param moving_average_n:
    :param x_offset:
    :param plotter_style:
    :return:
    """
    if plotter_style is None:
        plotter_style = PlotterStyle(300, (8, 4))
    y_label = f"roughness (moving average n = {moving_average_n})"
    data_dict = {}
    for key, sticher in sticher_dict.items():
        data_dict[key] = {"pixel_width": sticher.pixel_width, "data": sticher.stiched_data}

    result = _create_rms_figure(data_dict, moving_average_n, x_offset, plotter_style, y_label, subtract_average=True)
    return result


def create_gradient_rms_figure(sticher_dict: Dict[str, GDEFSticher], cutoff_percent=8, moving_average_n=1,
                               x_offset=0, plotter_style: PlotterStyle = None) -> Figure:
    """
    Creates a matplotlib figure, showing a graph of the root meean square of the gradient of the GDEFSticher objects in
    data_dict. The key value in data_dict is used as label in the legend.
    :param sticher_dict:
    :param cutoff_percent:
    :param moving_average_n:
    :param x_offset:
    :param plotter_style:
    :return:
    """
    if plotter_style is None:
        plotter_style = PlotterStyle(300, (8, 4))
    y_label = f"roughness(gradient) (moving average n = {moving_average_n})"

    data_dict = {}
    for key, sticher in sticher_dict.items():
        gradient_data = create_absolute_gradient_array(sticher.stiched_data, cutoff_percent / 100.0)
        data_dict[key] = {"pixel_width": sticher.pixel_width, "data": gradient_data}
    result = _create_rms_figure(data_dict, moving_average_n, x_offset, plotter_style, y_label)
    return result


def get_mu_sigma(values2d: np.ndarray) -> Tuple:
    """
    Returns mean and standard deviation of valus in values2d.
    :param values2d:
    :return:
    """
    z_values = values2d.flatten()
    z_values = z_values[~np.isnan(z_values)]
    return norm.fit(z_values)


def get_mu_sigma_moving_average(values2d: np.ndarray, moving_average_n=200, step=1) -> Tuple[List[float], List[float]]:
    """
    Calculate mu and sigma values as moving average, averaging over moving_average_n data-columns.
    :param values2d:
    :param n_bins:
    :return:
    """
    mu_list = []
    sigma_list = []

    for i in range(0, values2d.shape[1] - moving_average_n, step):
        mu, sigma = get_mu_sigma(values2d[:, i:i + moving_average_n])
        mu_list.append(mu)
        sigma_list.append(sigma)
    return mu_list, sigma_list


def set_z_histogram_from_ndarray_to_ax(ax, values2d: np.ndarray, title="", n_bins=200, units="µm"):
    """
     Also accepts a list of np.ndarray data (for plotting several histograms stacked)
     :param values2d:
     :param title:
     :param n_bins:
     param units: Can be set tu µm or nm (default is µm).
     :return:
     """
    if units == "nm":
        unit_factor = 1e9
        unit_label = "nm"
    else:
        unit_factor = 1e6
        unit_label = "\u03BCm"

    if isinstance(values2d, list):
        values2d_list = values2d
    else:
        values2d_list = [values2d]
    colors = []
    z_values_list = []
    best_filt_lines = []
    norm_bins_list = []
    for values2d in values2d_list:
        z_values = values2d.flatten()
        z_values = z_values[
            ~np.isnan(z_values)]  # remove all NaN values (~ is bitwise NOT opperator - similar to numpy.logical_not)
        z_values = z_values * unit_factor  # m -> µm/nm
        mu, sigma = norm.fit(z_values)
        norm_bins = np.linspace(z_values.min(), z_values.max(), 100)
        best_fit_line = norm.pdf(norm_bins, mu, sigma)
        z_values_list.append(z_values)
        best_filt_lines.append(best_fit_line)
        norm_bins_list.append(norm_bins)
        if len(colors) % 2 > 0:
            colors.append("red")
        else:
            colors.append("black")

    for i in range(len(z_values_list)):
        _, _, patch = ax.hist(z_values_list[i], density=True, bins=n_bins, edgecolor=colors[i], lw=1,
                              fc=to_rgba(colors[i], alpha=0.3), rwidth=1, histtype="bar", fill=True)  # color=colors[i]
        # plt.setp(patch, edgecolor=to_rgba(colors[i], alpha=1), lw=2)

    # # bars side by side:
    # _, _, patches = ax.hist(z_values_list, density=True, bins=n_bins, color=colors, rwidth=1, histtype="bar", fill=False)#
    # for i, patch in enumerate(patches):
    #     plt.setp(patch, edgecolor=colors[i])  # , lw=2)
    for i, line in enumerate(best_filt_lines):
        ax.plot(norm_bins_list[i], line, c=colors[i])
    ax.set_xlabel(f'z [{unit_label}]')
    ax.set_ylabel('Normalized counts')
    # ax.grid(True)
    if title:
        ax.set_title(f"{title}")
    else:
        ax.set_title(f"\u03BC={mu:.2f}, \u03C3={sigma:.2f}")

    return ax


def create_z_histogram_from_ndarray(values2d: np.ndarray, title="", n_bins=200, figure_size=(6, 3)):
    """
    Also accepts a list of np.ndarray data (for plotting several histograms stacked)
    :param values2d:
    :param title:
    :param n_bins:
    :return:
    """
    result, ax = plt.subplots(1, 1, figsize=figure_size, tight_layout=True, dpi=300)
    set_z_histogram_from_ndarray_to_ax(ax, values2d, title, n_bins)
    return result

    # graph_styler = plotter_style.graph_styler
    #
    # result, ax_compare_gradient_rms = plt.subplots(1, figsize=plotter_style.figure_size, dpi=plotter_style.dpi)
    #
    # ax_compare_gradient_rms.set_xlabel("[µm]")
    # ax_compare_gradient_rms.set_ylabel(
    #     f"roughness(gradient) (moving average n = {moving_average_n})")
    # ax_compare_gradient_rms.set_yticks([])
    # counter = 0
    # for key in data_dict:
    #     sticher = data_dict[key]
    #
    #     absolute_gradient_array = create_absolute_gradient_array(sticher.stiched_data, cutoff_percent / 100.0)
    #     x_pos, y_gradient_rms = create_xy_rms_data(absolute_gradient_array, sticher.pixel_width,
    #                                                moving_average_n)
    #     x_pos = [x + x_offset for x in x_pos]
    #     ax_compare_gradient_rms.plot(x_pos, y_gradient_rms, **graph_styler.dict, label=key)
    #     graph_styler.next_style()
    #
    #     ax_compare_gradient_rms.legend()
    # # fig.suptitle(f"cutoff = {cutoff_percent}%")
    # result.tight_layout()

# def create_surface_plot(values: np.ndarray, pixel_width, max_figure_size=(4, 4), dpi=96) -> Optional[Figure]:
#     def set_surface_to_axes(ax: Axes):
#         extent = extent_for_plot(values.shape, pixel_width)
#         im = ax.imshow(values * 1e9, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
#         # ax.set_title(self.comment)  # , pad=16)
#         ax.set_xlabel("µm", labelpad=1.0)
#         ax.set_ylabel("µm", labelpad=1.0)
#
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cax.set_title("nm", y=1)  # bar.set_label("nm")
#         plt.colorbar(im, cax=cax)
#
#     def extent_for_plot(shape, pixel_width):
#         width_in_um = shape[1] * pixel_width * 1e6
#         height_in_um = shape[0] * pixel_width * 1e6
#         return [0, width_in_um, 0, height_in_um]
#
#     if values is None:
#         return
#
#     figure_max, ax = plt.subplots(figsize=max_figure_size, dpi=dpi)
#     set_surface_to_axes(ax)
#     figure_max.tight_layout()
#
#     tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
#     figure_tight, ax = plt.subplots(figsize=tight_bbox.size, dpi=dpi)
#     set_surface_to_axes(ax)
#
#     return figure_tight


# def create_summary_figure(measurements: List[GDEFMeasurement], figure_size=(16, 10)):
#     n = len(measurements)
#     if n == 0:
#         return plt.subplots(1, figsize=figure_size, dpi=300)
#
#     optimal_ratio = figure_size[0] / figure_size[1]
#     dummy_fig = measurements[0].create_plot()
#     single_plot_ratio = dummy_fig.get_figwidth() / dummy_fig.get_figheight()
#     optimal_ratio /= single_plot_ratio
#
#     possible_ratios = []
#     for i in range(1, n+1):
#         for j in range(1, n+1):
#             if i*j >= n:
#                 x, y = i, j
#                 possible_ratios.append((x, y))
#                 break
#
#     # sort ratios by best fit to optimal ratio:
#     possible_ratios[:] = sorted(possible_ratios, key=lambda ratio: abs(ratio[0] / ratio[1] - optimal_ratio))
#     best_ratio = possible_ratios[0][1], possible_ratios[0][0]
#
#     result, ax_list = plt.subplots(*best_ratio, figsize=figure_size, dpi=300)
#     for i, measurement in enumerate(measurements):
#         y = i // best_ratio[0]
#         x = i - (y * best_ratio[0])
#         if best_ratio[1] > 1:
#             measurement.set_topography_to_axes(ax_list[x, y])
#         else:
#             measurement.set_topography_to_axes(ax_list[x])
#     i = len(measurements)
#     while i < best_ratio[0]*best_ratio[1]:
#         y = i // best_ratio[0]
#         x = i - (y * best_ratio[0])
#         ax_list[x, y].set_axis_off()
#         i += 1
#     result.tight_layout()
#     return result
