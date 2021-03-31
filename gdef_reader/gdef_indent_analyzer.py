"""
@author: Nathanael Jöhrmann
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from gdef_reader.gdef_measurement import GDEFMeasurement


class GDEFIndentAnalyzer:
    # speed optimization to reduce calculationtime of _is_pixel_in_radius()
    pixel_radius_distance_matrix = {}
    max_pixel_radius_value = 0

    def __init__(self, measurement: GDEFMeasurement):
        self.measurement = measurement
        self.radius = 0
        self.pileup = []
        self.indent = []
        self.surface = []

        self.below_surface_limit = 0
        self.above_surface_limit = 0

    @classmethod
    def _check_pixel_radius_dict(cls):
        if cls.pixel_radius_distance_matrix == {}:  # todo: or GDEFMeasurement.max_pixel_radius_value < self.settings...
            for x in range(-255, 256):
                for y in range(-255, 256):  # todo: symmetry???
                    cls.pixel_radius_distance_matrix[x, y] = (x ** 2 + y ** 2) ** 0.5

    def _is_pixel_in_radius(self, position, center, radius):
        """Radius in [m]"""
        # optimized for speed; do not introduce help variables, if not necessary
        if self.measurement.settings._pixel_width * GDEFIndentAnalyzer.pixel_radius_distance_matrix[
                (position[0] - center[0]), (position[1] - center[1])] <= radius:
            return True
        else:
            return False

    def _calc_volume_with_radius(self):
        self._check_pixel_radius_dict()
        minimum = np.min(self.measurement.values)
        if minimum is None:
            return 0
        self.radius = abs(7 * minimum)
        minimum_position = self.measurement._get_minimum_position()
        pixel_area = self.measurement.settings.pixel_area()
        result = 0
        for index, value in np.ndenumerate(self.measurement.values):
            if self._is_pixel_in_radius(index, minimum_position, self.radius):
                result += value * pixel_area
        return result

    def _get_indent_pile_up_area_mask(self, roughness_part=0.05):
        self._check_pixel_radius_dict()
        minimum = np.min(self.measurement.values)
        self.radius = abs(7 * minimum)
        minimum_position = self.measurement._get_minimum_position()

        self.below_surface_limit = roughness_part * minimum
        self.above_surface_limit = abs(self.below_surface_limit)

        result = np.zeros((self.measurement.values.shape[0], self.measurement.values.shape[1], 4))

        for index, _ in np.ndenumerate(self.measurement.values):
            if self._is_pixel_in_radius(index, minimum_position, self.radius):
                if self.measurement.values[index] < self.below_surface_limit:
                    result[index] = (0, 0, 1, 0.6)
                    self.indent.append(index)
                elif self.measurement.values[index] > self.above_surface_limit:
                    result[index] = (0, 1, 0, 0.6)
                    self.pileup.append(index)
                    self.surface.append(index)
                else:
                    result[index] = (0, 0, 0, 0.1)
        return result

    def add_indent_pile_up_mask_to_axes(self, ax: Axes, roughness_part=0.05) -> Axes:
        data = self._get_indent_pile_up_area_mask(roughness_part=roughness_part)
        extent = self.measurement.settings.size_in_um_for_plot()
        ax.imshow(data, cmap=plt.cm.Reds_r, interpolation='none', extent=extent)
        return ax

    # def analyze_indent(self, roughness_part=0.05):
        pass

    def _calc_area_and_volume(self, index_list):
        area = 0
        volume = 0
        for index in index_list:
            area += self.measurement.settings.pixel_area()
            volume += self.measurement.settings.pixel_area() * abs(self.measurement.values[index])
        return area, volume

    def get_summary_table_data(self):  # todo: consider move method to utils.py
        indent_area, indent_volume = self._calc_area_and_volume(self.indent)
        pileup_area, pileup_volume = self._calc_area_and_volume(self.pileup)

        result = [["z min / max [m]", f"{self.measurement.values.min():.2e} / {self.measurement.values.max():.2e}"]]
        # result.append(["maximum [m]", f"{self.measurement.values.max():.2e}"])
        result.append(["radius [m]", f"{self.radius:.2e}"])
        result.append(["surface limit [m]", f"+/- {self.above_surface_limit:.2e}"])

        result.append(["indent area [m^2]", f"{indent_area:.2e}"])
        result.append(["indent volume [m^3]", f"{indent_volume:.2e}"])

        result.append(["pileup area [m^2]", f"{pileup_area:.2e}"])
        result.append(["pileup volume [m^3]", f"{pileup_volume:.2e}"])

        return result
