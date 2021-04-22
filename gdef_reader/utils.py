"""
@author: Nathanael Jöhrmann
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Literal

import numpy as np
# todo: optional import:
from pptx_tools.creator import PPTXCreator
from pptx_tools.templates import AbstractTemplate
from scipy.stats import norm

from gdef_reporter.pptx_styles import summary_table, position_2x2_00, position_2x2_10, position_2x2_01, \
    minimize_table_height, position_2x2_11

if TYPE_CHECKING:
    from afm_tools.gdef_indent_analyzer import GDEFIndentAnalyzer
    from gdef_reader.gdef_importer import GDEFImporter
    from gdef_reader.gdef_measurement import GDEFMeasurement


def unit_factor_and_label(units: Literal["µm", "nm"]) -> tuple[float, str]:
    units_dict = {
        "nm": (1e9, "nm"),
        "µm": (1e6, "\u03BCm")
    }
    _units = units.replace("\u03BC", "µ")  # \u03BC is not equal to µ!
    return units_dict[_units]


def create_pygdf_files(input_path: Path, output_path: Path = None, create_images: bool = False) -> list[Path]:
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


def load_pygdf_measurements(path: Path) -> list[GDEFMeasurement]:
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

# todo: move to ... ???
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


# todo: move to ... ???
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


def create_xy_rms_data(values: np.ndarray, pixel_width, moving_average_n=1, subtract_average=True,
                       units: Literal["µm", "nm"] = "µm") -> tuple[list, list]:
    """
    :param values: 2D array
    :param pixel_width:
    :param moving_average_n:
    :param subtract_average:
    :param units:
    :return: (x_pos, y_rms)
    """
    factor, _ = unit_factor_and_label(units)
    x_pos = []
    y_rms = []
    for i in range(values.shape[1] - moving_average_n):
        x_pos.append((i + max(moving_average_n - 1, 0) / 2.0) * pixel_width * factor)
        y_rms.append(nanrms(values[:, i:i + moving_average_n], subtract_average=subtract_average))
    return x_pos, y_rms


def get_mu_sigma(values2d: np.ndarray) -> tuple:
    """
    Returns mean and standard deviation of valus in values2d.
    :param values2d:
    :return:
    """
    z_values = values2d.flatten()
    z_values = z_values[~np.isnan(z_values)]
    return norm.fit(z_values)


def get_mu_sigma_moving_average(values2d: np.ndarray, moving_average_n=200, step=1) -> tuple[list[float], list[float]]:
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
