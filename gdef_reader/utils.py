import pickle
from pathlib import Path
from typing import List, Optional

from pptx_tools.templates import AbstractTemplate

from gdef_reader.gdef_measurement import GDEFMeasurement

# todo: optional import:
from pptx_tools.creator import PPTXCreator, PPTXPosition


def load_pygdf_measurements(path: Path) -> List[GDEFMeasurement]:
    result = []
    files = path.rglob("*.pygdf")

    for filename in files:
        print(filename)
        with open(filename, 'rb') as file:
            measurement = pickle.load(file)
            measurement.filename = filename
            result.append(measurement)
    return result


def create_pptx_for_nanoindents(path, pptx_filename, pptx_template: Optional[AbstractTemplate] = None):
    pptx = PPTXCreator(template=pptx_template)
    pptx.add_title_slide(f"AFM on Nanoindents - {path.stem}")
    measurements = load_pygdf_measurements(path)
    for measurement in measurements:
        print(measurement.comment)
        slide = pptx.add_slide(measurement.comment)
        figure = measurement.create_plot()
        if figure is None:
            continue
        pptx.add_matplotlib_figure(figure, slide, PPTXPosition(0.1, 0.2))
        # figure.savefig(f"{measurement.filename.with_suffix('.png')}")  # , transparent=transparent)
        measurement.add_indent_pile_up_mask_to_axes(figure.axes[0])
        # figure.savefig(f"{measurement.filename.with_name(measurement.filename.stem + '_masked.png')}", dpi=96)
        pptx.add_matplotlib_figure(figure, slide, PPTXPosition(0.55, 0.2))
        figure.clear()
    pptx.save(path.joinpath(f"{pptx_filename}.pptx"))
