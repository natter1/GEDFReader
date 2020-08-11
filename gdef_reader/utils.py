import copy
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
# todo: optional import:
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
    result = np.gradient(array2d)  # [0]
    result = np.sqrt(result[0] ** 2 + result[1] ** 2)
    max_grad = np.nanmax(result)
    with np.nditer(result, op_flags=['readwrite']) as it:
        for x in it:
            if not np.isnan(x) and x > cutoff * max_grad:
                x[...] = np.nan
    return result


def create_image_data(array2d):
    data_min = np.nanmin(array2d)
    array2d = (array2d - min(0, data_min)) / (np.nanmax(array2d) - min(0, data_min))  # normalize the data to 0 - 1
    array2d = 255 * array2d  # Now scale by 255
    return array2d.astype(np.uint8)

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


def create_rms_per_column_figure(values: np.ndarray, pixel_width, title=None, moving_average_n=1) -> Figure:
    """

    :param values: 2D array
    :param pixel_width: in meter
    :param title: optional figure title
    :param moving_average_n: number of columns for moving average
    :return: matplotlib Figure
    """
    x_pos, y_rms = create_xy_rms_data(values, pixel_width, moving_average_n)
    result, (ax_rms) = plt.subplots(1, 1, figsize=(10, 6))
    ax_rms.plot(x_pos, y_rms, 'r')
    ax_rms.set_xlabel("[µm]")
    ax_rms.set_ylabel(f"root mean square (moving average over {moving_average_n} column(s))")
    if title:
        result.suptitle(f'{title}', fontsize=16)
    return result


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


def create_absolute_gradient_figure(values: np.ndarray, cutoff_percent_list, nan_color='red') -> Figure:
    result, ax_list_cutoff = plt.subplots(len(cutoff_percent_list), 1, figsize=(len(cutoff_percent_list) * 0.4, 13))

    cmap_gray_red_nan = copy.copy(plt.cm.gray)  # use copy to prevent unwanted changes to other plots somewhere else
    cmap_gray_red_nan.set_bad(color=nan_color)

    for i, percent in enumerate(cutoff_percent_list):
        absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
        ax_list_cutoff[i].imshow(absolut_gradient_array, cmap_gray_red_nan)
        ax_list_cutoff[i].set_title(f'gradient cutoff {percent}%')
        ax_list_cutoff[i].set_axis_off()
    return result


def create_absolute_gradient_rms_figure(values: np.ndarray, cutoff_percent_list, pixel_width, moving_average_n=1) -> Figure:
    result, (ax_gradient_rms) = plt.subplots(1, 1, figsize=(10, 10))
    ax_gradient_rms.set_xlabel("[µm]")
    ax_gradient_rms.set_ylabel(f"absolute gradient root mean square (moving average over {moving_average_n} column(s))")

    for i, percent in enumerate(cutoff_percent_list):
        absolut_gradient_array = create_absolute_gradient_array(values, percent / 100.0)
        x_pos, y_gradient_rms = create_xy_rms_data(absolut_gradient_array, pixel_width, moving_average_n)
        ax_gradient_rms.plot(x_pos, y_gradient_rms, label=f"{percent}%")
    ax_gradient_rms.legend()
    return result
