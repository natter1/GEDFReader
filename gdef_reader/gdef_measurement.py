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

    def pixel_width(self):
        return self.max_width / self.columns

    def pixel_area(self):
        return self.pixel_width()**2

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

        self.gdf_filename = ""  # filename of original *.gdf file
        self.filename: Optional[Path] = None  # filename of pickled *.pygdf

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

    def save_png(self, filename, max_figure_size=(6, 6), dpi=300, transparent=False):
        figure = self.create_plot(max_figure_size=max_figure_size, dpi=dpi)
        if figure:
            figure.savefig(filename, transparent=transparent)

    def _get_minimum_position(self):
        minimum = np.min(self.values)
        minimum_position = (0, 0)
        for index, value in np.ndenumerate(self.values):
            if value == minimum:
                minimum_position = index
        return minimum_position

    def _is_pixel_in_radius(self, position, center, radius):
        """Radius in [m]"""
        pixel_length = self.settings.pixel_width()
        distance_pixel = ((position[0]-center[0])**2 + (position[1]-center[1])**2)**0.5
        if pixel_length*distance_pixel <= radius:
            return True
        else:
            return False

    def _calc_volume_with_radius(self):
        minimum = np.min(self.values)
        if minimum is None:
            return 0
        radius = abs(7 * minimum)
        minimum_position = self._get_minimum_position()
        pixel_area = self.settings.pixel_area()
        result = 0
        for index, value in np.ndenumerate(self.values):
            if self._is_pixel_in_radius(index, minimum_position, radius):
                result += value * pixel_area
        return result

    def _get_indent_pile_up_area_mask(self, roughness_part=0.05):
        minimum = np.min(self.values)
        radius = abs(7 * minimum)
        minimum_position = self._get_minimum_position()

        result = np.zeros((self.values.shape[0], self.values.shape[1], 4))
        for index, _ in np.ndenumerate(self.values):
            if self._is_pixel_in_radius(index, minimum_position, radius):
                if self.values[index] < roughness_part * minimum:
                    result[index] = (0, 0, 1, 0.6)
                elif self.values[index] > roughness_part * abs(minimum):
                    result[index] = (0, 1, 0, 0.6)
                else:
                    result[index] = (0, 0, 0, 0.1)
        return result

    def _get_greyscale_data(self):
        # Normalised [0,1]
        data_min = np.min(self.values)
        data_ptp = np.ptp(self.values)

        result = np.zeros((self.values.shape[0], self.values.shape[1], 4))
        for (nx, ny), _ in np.ndenumerate(self.values):
            value = (self.values[nx, ny] - data_min) / data_ptp
            result[nx, ny] = (value, value, value, 0)
        return result

    def create_plot(self, max_figure_size=(6, 6), dpi=300) -> Optional[Figure]:
        def create_figure(data, figure_size):
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            im = ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
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
        size = (tight_bbox.width * 1.25, tight_bbox.height)  # Legend takes 20% of width -> 100%/80% = 1.25
        figure_tight, ax, im = create_figure(self.values * 1e9, size)

        bar = figure_tight.colorbar(im, ax=ax)  # shrink=(1-0.15-0.05))  # 0.15 - fraction; 0.05 - pad
        bar.ax.set_title("nm")  # bar.set_label("nm")

        return figure_tight  # , ax

    def add_indent_pile_up_mask_to_axes(self, ax: Axes) -> Axes:
        data = self._get_indent_pile_up_area_mask()
        extent = self.settings.extent_for_plot()
        ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
        return ax

    def correct_background(self):
        """Set average value to zero and subtract tilted background-plane."""
        self._do_median_level(subtract_mean_plane=True)

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
