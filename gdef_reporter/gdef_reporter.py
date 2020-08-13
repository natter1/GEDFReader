import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union

from pptx_tools.creator import PPTXCreator
from pptx_tools.position import PPTXPosition
from pptx_tools.table_style import PPTXTableStyle

from gdef_reader.etit169_pptx_template import TemplateETIT169
from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_measurement import GDEFMeasurement

import matplotlib.pyplot as plt

from gdef_reader.pptx_styles import summary_table, minimize_table_height


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

    # def set_filter_ids(self, ids: List[int]):
    #     self.filter_ids = ids

    @property
    def filtered_measurements(self) -> List[GDEFMeasurement]:
        return [x for x in self.measurements if not x.gdf_block_id in self.filter_ids]

    # def save(self, path: Path):
    #     if path:
    #         path.mkdir(parents=True, exist_ok=True)
    #         filename = path.joinpath(f"{self.basename}.pygdf")
    #     with open(filename, 'wb') as file:
    #         pickle.dump(self, file, 3)


class GDEFReporter:
    def __init__(self, gdf_containers: Union[GDEFContainer, List[GDEFContainer]] = None):
        if isinstance(gdf_containers, GDEFContainer):
            self.gdf_containers = [gdf_containers]
        else:
            self.gdf_containers: List[GDEFContainer] = gdf_containers
        self.primary_gdf_folder = gdf_containers[0].path.parent  # todo: check for muliple folders
        self.pptx = None

    def create_summary_pptx(self, pptx_template=TemplateETIT169()):
        self.pptx = PPTXCreator(template=pptx_template)
        title_slide = self.pptx.add_title_slide(f"AFM - {self.primary_gdf_folder.stem}")

        table_data = [["file", "date"]]
        for container in self.gdf_containers:
            table_data.append([f"{container.path.stem}.gdf", container.last_modification_datetime])

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
            else:
                measurement.set_topography_to_axes(ax_list[x])
        i = len(measurements)
        while i < best_ratio[0] * best_ratio[1]:
            y = i // best_ratio[0]
            x = i - (y * best_ratio[0])
            ax_list[x, y].set_axis_off()
            i += 1
        result.tight_layout()
        return result

# def creation_date(path_to_file):
#     """
#     Try to get the date that a file was created, falling back to when it was
#     last modified if that isn't possible.
#     See http://stackoverflow.com/a/39501288/1709587 for explanation.
#     """
#     if platform.system() == 'Windows':
#         return time.ctime(os.path.getctime(path_to_file))
#     else:
#         stat = os.stat(path_to_file)
#         try:
#             return time.ctime(stat.st_birthtime)
#         except AttributeError:
#             # We're probably on Linux. No easy way to get creation dates here,
#             # so we'll settle for when its content was last modified.
#             return time.ctime(stat.st_mtime)