import copy
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# todo: optional import:
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pptx_tools.creator import PPTXCreator
from pptx_tools.templates import AbstractTemplate

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_indent_analyzer import GDEFIndentAnalyzer
from gdef_reader.gdef_measurement import GDEFMeasurement
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
            measurement.filename = filename
            result.append(measurement)
    return result


def create_png_for_nanoindents(path: Path, png_save_path: Optional[Path]= None):
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
        print(png_save_path.joinpath(f"{measurement.filename.stem + '.png'}"))
        figure.savefig(png_save_path.joinpath(f"{measurement.filename.stem + '.png'}"), dpi=96)  # , transparent=transparent)
        indent_analyzer.add_indent_pile_up_mask_to_axes(figure.axes[0])
        print(png_save_path.joinpath(f"{measurement.filename.stem + '_masked.png'}"))
        figure.savefig(png_save_path.joinpath(f"{measurement.filename.stem + '_masked.png'}"), dpi=96)
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
        table_shape = pptx.add_table(slide, measurement.get_summary_table_data(), position_2x2_01(), table_style=summary_table())
        minimize_table_height(table_shape)
        # figure.savefig(f"{measurement.basename.with_suffix('.png')}")  # , transparent=transparent)

        indent_analyzer.add_indent_pile_up_mask_to_axes(figure.axes[0], roughness_part=0.05)
        # figure.savefig(f"{measurement.basename.with_name(measurement.basename.stem + '_masked.png')}", dpi=96)
        pptx.add_matplotlib_figure(figure, slide, position_2x2_10())
        table_shape = pptx.add_table(slide, indent_analyzer.get_summary_table_data(), position_2x2_11(), table_style=summary_table())
        minimize_table_height(table_shape)
        figure.clear()
    pptx.save(path.joinpath(f"{pptx_filename}.pptx"), overwrite=True)  # todo: remove overwrite=True when testing is finished


# get rms roughness
def nanrms(x: np.ndarray, axis=None):
    """Returns root mean square of given numpy.ndarray x."""
    return np.sqrt(np.nanmean(x**2, axis=axis))


def create_absolute_gradient_array(array2d, cutoff = 1.0):
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

def create_xy_rms_data(values: np.ndarray, pixel_width, moving_average_n=1) -> Tuple[list, list]:
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
        y_rms.append(nanrms(values[:, i:i + moving_average_n]))
    return x_pos, y_rms


def save_figure(figure: Figure, output_path: Path, filename: str,  png: bool = True, pdf: bool = False) -> None:
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


def create_absolute_gradient_figures(values: np.ndarray, cutoff_percent_list, nan_color='red') -> Figure:
    result, ax_list_cutoff = plt.subplots(len(cutoff_percent_list), 1, figsize=(len(cutoff_percent_list) * 0.4, 13))

    cmap_gray_red_nan = copy.copy(plt.cm.gray)  # use copy to prevent unwanted changes to other plots somewhere else
    cmap_gray_red_nan.set_bad(color=nan_color)

    for i, percent in enumerate(cutoff_percent_list):
        absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
        ax_list_cutoff[i].imshow(absolut_gradient_array, cmap_gray_red_nan)
        ax_list_cutoff[i].set_title(f'gradient cutoff {percent}%')
        ax_list_cutoff[i].set_axis_off()
    return result


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
