import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    def pixel_width(self):
        return self._pixel_width  # self.max_width / self.columns

    def pixel_area(self):
        return self.pixel_width**2

    def extent_for_plot(self):
        width_in_um = self.max_width * 1e6
        height_in_um = self.max_height * (self.lines - self.missing_lines) / self.lines * 1e6
        return [0, width_in_um, 0, height_in_um]

    def shape(self):
        return self.columns - self.missing_lines, self.lines


class GDEFMeasurement:
    def __init__(self):
        self.header: Optional[GDEFHeader] = None
        self.spm_image_file_vesion = None

        self.settings = GDEFSettings()

        self._values_original = None  # do not change!
        self.values = None
        self.preview = None
        self.comment = ''

        self.gdf_filename = ""  # basename of original *.gdf file
        self.filename: Optional[Path] = None  # basename of pickled *.pygdf

        self.background_corrected = False

    @property
    def values_original(self):
        return self._values_original

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, 3)

    @staticmethod
    def load(filename):
        with open(filename, 'rb'):
            return pickle.load(filename)

    def save_png(self, filename, max_figure_size=(4, 4), dpi=300, transparent=False):
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

    def create_plot(self, max_figure_size=(4, 4), dpi=96) -> Optional[Figure]:
        def create_figure(data, figure_size):
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            im = ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
            #fig.suptitle(self.comment + f" {self.settings.scan_speed*1e6:.0f} µm/s")
            ax.set_title(self.comment[12:] + f" {self.settings.scan_speed*1e6:.0f} µm/s")
            ax.set_xlabel("µm")
            ax.set_ylabel("µm")
            return fig, ax, im

        if self.values is None:
            return

        if self.settings.source_channel != 11:
            return  # for now, only plot topography (-> soutce_channel == 11)

        # self._do_median_level()

        extent = self.settings.extent_for_plot()

        figure_max, ax, im = create_figure(self.values * 1e9, max_figure_size)
        tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
        size = (tight_bbox.width * 1.25, tight_bbox.height *1.05)  # Legend takes 20% of width -> 100%/80% = 1.25
        figure_tight, ax, im = create_figure(self.values * 1e9, size)
        bar = figure_tight.colorbar(im, ax=ax)  # shrink=(1-0.15-0.05))  # 0.15 - fraction; 0.05 - pad
        bar.ax.set_title("nm")  # bar.set_label("nm")

        return figure_tight  # , ax

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
        result.append(["basename", f"{self.filename.stem}"])

        return result