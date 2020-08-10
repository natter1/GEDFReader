import pickle
from pathlib import Path
from typing import List, Optional

# todo: optional import:
from pptx_tools.creator import PPTXCreator
from pptx_tools.templates import AbstractTemplate

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_indent_analyzer import GDEFIndentAnalyzer
from gdef_reader.gdef_measurement import GDEFMeasurement
from gdef_reader.pptx_styles import summary_table, position_2x2_00, position_2x2_10, position_2x2_01, \
    minimize_table_height, position_2x2_11


def create_pygdf_files(input_path: Path, output_path: Path = None, create_images: bool = False):
    gdf_filenames = input_path.glob("*.gdf")  # glob returns a generator, so gdf_filenames can only be used once!

    if not output_path:
        output_path = input_path.joinpath("pygdf")
        output_path.mkdir(parents=True, exist_ok=True)

    gdf_measurements = []
    for gdf_filename in gdf_filenames:
        print(gdf_filename)
        gdf_importer = GDEFImporter(gdf_filename)
        pygdf_path = output_path.joinpath(gdf_filename.stem)
        gdf_measurements.append(gdf_importer.export_measurements(pygdf_path, create_images))


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
