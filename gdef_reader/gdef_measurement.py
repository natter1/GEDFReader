import pickle
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gdef_reader.gdef_data_strucutres import GDEFHeader


class GDEFSettings:
    def __init__(self):
        # Settings:
        self.lines = None
        self.columns = None
        self.missing_lines = None
        self.line_mean = None
        self.line_mean_order = None
        self.invert_line_mean = None
        self._plane_corr = None
        self.invert_plane_corr = None
        self.max_width = None
        self.max_height = None
        self.offset_x = None
        self.offset_y = None
        self.z_unit = None
        self.retrace = None
        self.z_linearized = None
        self.scan_mode = None
        self.z_calib = None
        self.x_calib = None
        self.y_calib = None
        self.scan_speed = None
        self.set_point = None
        self.bias_voltage = None
        self.loop_gain = None
        self.loop_int = None
        self.phase_shift = None
        self.scan_direction = None
        self.digital_loop = None
        self.loop_filter = None
        self.fft_type = None
        self.xy_linearized = None
        self.retrace_type = None
        self.calculated = None
        self.scanner_range = None
        self.pixel_blend = None
        self.source_channel = None
        self.direct_ac = None
        self.id = None
        self.q_factor = None
        self.aux_gain = None
        self.fixed_palette = None
        self.fixed_min = None
        self.fixed_max = None
        self.zero_scan = None
        self.measured_amplitude = None
        self.frequency_offset = None
        self.q_boost = None
        self.offset_pos = None

        self._pixel_width = None

    @property
    def pixel_width(self) -> float:
        return self._pixel_width  # self.max_width / self.columns

    def pixel_area(self) -> float:
        return self.pixel_width**2

    def size_in_um_for_plot(self) -> Tuple[float, float, float, float]:
        width_in_um = self.max_width * 1e6
        height_in_um = self.max_height * (self.lines - self.missing_lines) / self.lines * 1e6
        return [0, width_in_um, 0, height_in_um]

    def shape(self) -> Tuple[int, int]:
        return self.columns - self.missing_lines, self.lines


class GDEFMeasurement:
    def __init__(self):
        self.header: Optional[GDEFHeader] = None
        self.spm_image_file_vesion = None

        self.settings = GDEFSettings()

        self._values_original = None  # do not change! Use values instead
        self.values = None
        self.preview = None
        self.comment = ''

        self.gdf_basename = ""  # basename of original *.gdf file
        self.filename: Optional[Path] = None  # basename of pickled *.pygdf
        self.gdf_block_id = None

        self.background_corrected = False

    @property
    def name(self) -> str:
        return f"{self.gdf_basename}_block_{self.gdf_block_id:03}"

    @property
    def values_original(self) -> np.ndarray:
        return self._values_original

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, 3)

    @staticmethod
    def load(filename) -> "GDEFMeasurement":
        with open(filename, 'rb'):
            return pickle.load(filename)

    # todo: check possible types for filename (str, path, ...)
    def save_png(self, filename, max_figure_size=(4, 4), dpi: int = 300, transparent: bool = False):
        figure = self.create_plot(max_figure_size=max_figure_size, dpi=dpi)
        if figure:
            figure.savefig(filename, transparent=transparent, dpi=dpi)

    def _get_minimum_position(self):
        # ---------------------------------------------------------------------------------------------------
        # this makes code actually slower (not this method, but other code parts ^^); so do not use np.where:
        # delme = np.where(self.values == np.amin(self.values))
        # return delme[0][0], delme[1][0]
        # ---------------------------------------------------------------------------------------------------
        minimum = np.min(self.values)
        minimum_position = (0, 0)
        for index, value in np.ndenumerate(self.values):
            if value == minimum:
                minimum_position = index
                break
        return minimum_position

    def _get_greyscale_data(self):
        # Normalised [0,1]
        data_min = np.min(self.values)
        data_ptp = np.ptp(self.values)

        result = np.zeros((self.values.shape[0], self.values.shape[1], 4))
        for (nx, ny), _ in np.ndenumerate(self.values):
            value = (self.values[nx, ny] - data_min) / data_ptp
            result[nx, ny] = (value, value, value, 0)
        return result

    def set_topography_to_axes(self, ax: Axes):
        if self.values is None:
            ax.set_title(self.comment)
            print(f"GDEFMeasurement {self.name} has values==None")
            return
        # todo: refactor to extra method
        if self.settings.source_channel == 11:
            title = "topography"
            unit = "nm"
            values = self.values * 1e9  # m -> nm
        elif self.settings.source_channel == 9:
            title = "bending"
            unit = "N"
            values = self._values_original
        elif self.settings.source_channel == 12:
            title = "phase"
            unit = "deg"
            # factor 18.0 from gwyddion - seems to create too large values (e.g. 600 degree)
            values = self._values_original  # * 18.0
        else:
            title = f"SC: {self.settings.source_channel}"
            unit = f"SC: {self.settings.source_channel}"
            values = self._values_original

        extent = self.settings.size_in_um_for_plot()
        im = ax.imshow(values, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
        if self.comment:
            title = self.comment
        ax.set_title(title)  # , pad=16)
        ax.set_xlabel("µm", labelpad=1.0)
        ax.set_ylabel("µm", labelpad=1.0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if self.settings.source_channel in [11]:
            cax.set_title(unit, y=1)  # bar.set_label("nm")
        else:
            cax.set_title(unit, y=0, pad=-15) #-0.2)#1)  # bar.set_label("nm")
            cax.title.set_color("red")
        plt.colorbar(im, cax=cax)

    def create_plot(self, max_figure_size=(4, 4), dpi=96) -> Optional[Figure]:
        if self.values is None:
            return

        # if self.settings.source_channel != 11:
        #     return  # for now, only plot topography (-> source_channel == 11)

        figure_max, ax = plt.subplots(figsize=max_figure_size, dpi=dpi)
        self.set_topography_to_axes(ax)

        tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
        figure_tight, ax = plt.subplots(figsize=tight_bbox.size, dpi=dpi)
        self.set_topography_to_axes(ax)

        return figure_tight

    def correct_background(self):
        """Set average value to zero and subtract tilted background-plane."""
        if not self.background_corrected:
            self._do_median_level(subtract_mean_plane=True)
            self.background_corrected = True

    def _subtract_mean_plane(self):
        try:
            value_gradient = np.gradient(self.values)
        except ValueError:
            return
        mean_value_gradient_x = value_gradient[0].mean()
        mean_value_gradient_y = value_gradient[1].mean()
        for (nx, ny), _ in np.ndenumerate(self.values):
            self.values[nx, ny] = self.values[nx, ny] - nx * mean_value_gradient_x - ny * mean_value_gradient_y

    def _do_median_level(self, subtract_mean_plane: bool = True):
        if subtract_mean_plane:
            self._subtract_mean_plane()
        try:
            self.values = self.values - self.values.mean()
        except ValueError:
            pass

    def get_summary_table_data(self):  # todo: consider move method to utils.py
        result = [["source channel", self.settings.source_channel]]
        result.append(["retrace", self.settings.retrace])
        result.append(["missing lines", self.settings.missing_lines])
        result.append(["max width [m]", f"{self.settings.max_width:.2e}"])
        result.append(["max height [m]", f"{self.settings.max_height:.2e}"])
        result.append(["scan speed [µm/s]", f"{self.settings.scan_speed*1e6:.0f}"])
        if self.filename:
            result.append(["basename", f"{self.filename.stem}"])
        else:
            result.append(["name", f"{self.gdf_basename}_block_{self.gdf_block_id:03}"])

        return result