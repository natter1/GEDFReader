import os
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import png
from pptx_tools.creator import PPTXCreator
from pptx_tools.position import PPTXPosition
from pptx_tools.table_style import PPTXTableStyle

from gdef_reader.etit169_pptx_template import TemplateETIT169
from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_measurement import GDEFMeasurement
from gdef_reader.gdef_sticher import GDEFSticher
from gdef_reader.pptx_styles import summary_table, minimize_table_height

import numpy as np


# todo: class GDEFMeasurement_Collection


class GDEFContainer:
    """
    Container class for all measurements inside a *.gdf-file
    """
    def __init__(self, gdf_path: Path):
        self.basename: str = gdf_path.stem
        self.base_path_name = gdf_path.parent.stem
        self.path: Path = gdf_path
        self.last_modification_datetime: datetime = datetime.fromtimestamp(os.path.getmtime(gdf_path))
        self.measurements: List[GDEFMeasurement] = GDEFImporter(gdf_path).export_measurements()
        self.filter_ids: List[int] = []
        self.descriprion = f"{self.base_path_name} - {self.basename}"

    @property
    def filtered_measurements(self) -> List[GDEFMeasurement]:
        return [x for x in self.measurements if x.gdf_block_id not in self.filter_ids]

    def correct_backgrounds(self, use_gradient_plane: bool = True, legendre_deg: int = 1, keep_offset: bool = False):
        for measurement in self.measurements:
            measurement.correct_background(use_gradient_plane, legendre_deg, keep_offset)


class GDEFContainerList(list):
    """
    List of GDEFContainer objects and some helper methods
    """
    def __init__(self, containers: Union[GDEFContainer, List[GDEFContainer, None]] = None):
        if isinstance(containers, GDEFContainer):
            self.append(containers)
        elif containers:
            self.extend(containers)

    def correct_backgrounds(self, use_gradient_plane: bool = True, legendre_deg: int = 1, keep_offset: bool = False):
        for container in self:
            container.correct_backgrounds(use_gradient_plane, legendre_deg, keep_offset)

    def set_filter_ids(self, filter_dict: dict):
        for container in self:
            container.filter_ids = filter_dict[container.basename]


class GDEFReporter:
    def __init__(self, gdf_containers: Union[GDEFContainer, List[GDEFContainer], None] = None):
        self.gdf_containers: GDEFContainerList = GDEFContainerList(gdf_containers)
        self.primary_gdf_folder = gdf_containers[0].path.parent  # todo: check for multiple folders
        self.title = f"AFM - {self.primary_gdf_folder.stem}"
        self.subtitle = self.title
        self.pptx = None

    def create_summary_pptx(self, pptx_template=TemplateETIT169()):
        self.pptx = PPTXCreator(template=pptx_template)
        title_slide = self.pptx.add_title_slide(self.title)

        table_data = self.get_files_date_table_data()

        table_style = PPTXTableStyle()
        table_style.set_width_as_fraction(0.55)
        self.pptx.add_table(title_slide, table_data, PPTXPosition(0.0, 0.224, 0.1, 0.1), table_style)

        for container in self.gdf_containers:
            slide = self.pptx.add_slide(f"Overview - {container.basename}.gdf")
            self.pptx.add_matplotlib_figure(self.create_summary_figure(container.measurements), slide, PPTXPosition(0, 0.115), zoom=0.62)

            table_style = summary_table()
            table_style.font_style.set(size=11)
            table_style.set_width_as_fraction(0.245)
            table_data = container.measurements[0].get_summary_table_data()
            table_data.append(["comment", container.measurements[0].comment])
            table_shape = self.pptx.add_table(slide, table_data, PPTXPosition(0.75, 0.115), table_style)
            minimize_table_height(table_shape)

        return self.pptx

    def get_files_date_table_data(self):
        result = [["file", "date"]]
        for container in self.gdf_containers:
            result.append([f"{container.path.stem}.gdf", container.last_modification_datetime])
        return result

    def create_summary_figure(self, measurements: List[GDEFMeasurement], figure_size=(16, 10)):
        n = len(measurements)
        if n == 0:
            return plt.subplots(1, figsize=figure_size, dpi=300)

        optimal_ratio = figure_size[0] / figure_size[1]
        dummy_fig = measurements[0].create_plot()
        single_plot_ratio = dummy_fig.get_figwidth() / dummy_fig.get_figheight()
        optimal_ratio /= single_plot_ratio

        possible_ratios = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j >= n:
                    x, y = i, j
                    possible_ratios.append((x, y))
                    break

        # sort ratios by best fit to optimal ratio:
        possible_ratios[:] = sorted(possible_ratios, key=lambda ratio: abs(ratio[0] / ratio[1] - optimal_ratio))
        best_ratio = possible_ratios[0][1], possible_ratios[0][0]

        result, ax_list = plt.subplots(*best_ratio, figsize=figure_size, dpi=300)
        for i, measurement in enumerate(measurements):
            y = i // best_ratio[0]
            x = i - (y * best_ratio[0])
            if best_ratio[1] > 1:
                measurement.set_topography_to_axes(ax_list[x, y])
            elif best_ratio[0] > 1:
                measurement.set_topography_to_axes(ax_list[x])
            else:
                measurement.set_topography_to_axes(ax_list)
        i = len(measurements)
        while i < best_ratio[0] * best_ratio[1]:
            y = i // best_ratio[0]
            x = i - (y * best_ratio[0])
            ax_list[x, y].set_axis_off()
            i += 1
        result.tight_layout()
        return result

    @classmethod
    def create_stiched_data(cls, measurements, initial_x_offset_fraction=0.35, show_control_plots=False):
        values_list = []
        for measurement in measurements:
            values_list.append(measurement.values)
        return GDEFSticher(values_list, initial_x_offset_fraction, show_control_plots).stiched_data

    @classmethod
    def _create_image_data(cls, data: np.ndarray):
        """
        Transform given data array into a n array with uint8 values between 0 and 255.
        :param data:
        :return:
        """
        data_min = np.nanmin(data)
        data = (data - np.min([0, data_min])) / (np.nanmax(data) - np.min([0, data_min]))  # normalize the data to 0 - 1
        data = 255 * data  # Now scale by 255
        return data.astype(np.uint8)

    @classmethod
    def data_to_png(cls, data: np.ndarray, mode='L'):
        """
        Can be used to get a png-object for pptx or to save to hard disc.
        Mode 'L' means greyscale. Mode'LA' is greyscale with alpha channel.
        """
        image_data = cls._create_image_data(data)
        return png.from_array(image_data, mode=mode)  # .save(f"{samplename}_stiched.png")
