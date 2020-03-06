import pickle
from pathlib import Path
from typing import Optional, List, Tuple

# from gdef_reader.gdef_importer import GDEFHeader, GDEFControlBlock
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gdef_reader.gdef_data_strucutres import GDEFHeader

import matplotlib.pyplot as plt
import numpy as np

class GDEFMeasurement:
    def __init__(self):
        self.header: Optional[GDEFHeader] = None
        self.spm_image_file_vesion = None

        self.value = None
        self.preview = None
        self.comment = ''

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

        self.gdf_filename = ""  # filename of original *.gdf file
        self.filename: Optional[Path] = None  # filename of pickled *.pygdf

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, 3)

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(filename)

    def save_png(self, filename, max_figure_size=(6, 6), dpi=300, transparent=False):
        figure = self.create_plot(max_figure_size=max_figure_size, dpi=dpi)
        if figure:
            figure.savefig(filename, transparent=transparent)

    def _get_minimum_position(self):
        minimum = np.min(self.value)
        minimum_position = (0, 0)
        for index, value in np.ndenumerate(self.value):
            if value == minimum:
                minimum_position = index
        return minimum_position

    def _is_pixel_in_radius(self, position, center, radius):
        """Radius in [m]"""
        pixel_length = self.max_width / self.columns
        distance_pixel = ((position[0]-center[0])**2 + (position[1]-center[1])**2)**0.5
        if pixel_length*distance_pixel <= radius:
            return True
        else:
            return False

    def _calc_volume_with_radius(self):
        minimum = np.min(self.value)
        if minimum is None:
            return 0
        radius = abs(7 * minimum)
        minimum_position = self._get_minimum_position()
        pixel_area = (self.max_width / self.columns)**2
        result = 0
        for index, value in np.ndenumerate(self.value):
            if self._is_pixel_in_radius(index, minimum_position, radius):
                result += value * pixel_area
        return result

    def _get_indent_pile_up_area_mask(self):
        minimum = np.min(self.value)
        radius = abs(7 * minimum)
        minimum_position = self._get_minimum_position()
        roughness_part = 0.05

        result = np.zeros((self.value.shape[0], self.value.shape[1], 4))
        for index, _ in np.ndenumerate(self.value):
            if self._is_pixel_in_radius(index, minimum_position, radius):
                if self.value[index] < roughness_part * minimum:
                    result[index] = (0, 0, 1, 0.6)
                elif self.value[index] > roughness_part * abs(minimum):
                    result[index] = (0, 1, 0, 0.6)
                else:
                    result[index] = (0, 0, 0, 0.1)
        return result


    def _get_greyscale_data(self):
        # Normalised [0,1]
        data_min = np.min(self.value)
        data_ptp = np.ptp(self.value)

        result = np.zeros((self.value.shape[0], self.value.shape[1], 4))
        for (nx, ny), _ in np.ndenumerate(self.value):
            value = (self.value[nx, ny] - data_min) / data_ptp
            result[nx, ny] = (value, value, value, 0)
            # result[nx, ny] = (data[nx, ny], data[nx, ny], data[nx, ny])
        return result

    def _get_extend_for_plot(self):
        return [0, self.max_width * 1e6, 0, self.max_height * (self.lines - self.missing_lines) / self.lines * 1e6]

    def create_plot(self, max_figure_size=(6, 6), dpi=300) -> Optional[Tuple[Figure, Axes]]:
        def create_figure(data, extent, figure_size):
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            im = ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
            ax.set_xlabel("µm")
            ax.set_ylabel("µm")
            return fig, ax, im

        if self.value is None:
            return

        if self.source_channel != 11:
            return  # for now, only plot topography (-> soutce_channel == 11)

        self._do_median_level()

        extent = self._get_extend_for_plot()

        figure_max, ax, im = create_figure(self.value*1e9, extent, max_figure_size)
        tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
        size = (tight_bbox.width * 1.25, tight_bbox.height)  # Legend takes 20% of width -> 100%/80% = 1.25
        figure_tight, ax, im = create_figure(self.value * 1e9, extent, size)

        bar = figure_tight.colorbar(im, ax=ax)  # shrink=(1-0.15-0.05))  # 0.15 - fraction; 0.05 - pad
        bar.ax.set_title("nm") # bar.set_label("nm")

        return figure_tight, ax

    def add_indent_pile_up_mask_to_axes(self, ax: Axes) -> Axes:
        data = self._get_indent_pile_up_area_mask()
        extent = self._get_extend_for_plot()
        im = ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
        return Axes

    def _do_subtract_mean_plane(self):
        try:
            value_gradient = np.gradient(self.value)
        except ValueError:
            return
        mean_value_gradient_x = value_gradient[0].mean()
        mean_value_gradient_y = value_gradient[1].mean()
        for (nx, ny), _ in np.ndenumerate(self.value):
            self.value[nx, ny] = self.value[nx, ny] - nx * mean_value_gradient_x - ny * mean_value_gradient_y

    def _do_median_level(self, subtract_mean_plane: bool = True):
        if subtract_mean_plane:
            self._do_subtract_mean_plane()
        try:
            self.value = self.value - self.value.mean()
        except ValueError:
            pass
