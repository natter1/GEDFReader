"""
A *.gdf file can contain many AFM measurements. To handle a single measurement the class GDEFMeasurement is used.
All the settings used during that specific measurement are stored in a GDEFSettings object.
@author: Nathanael Jöhrmann
"""
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from afm_tools.background_correction import BGCorrectionType, \
    correct_background
from gdef_reader.gdef_data_strucutres import GDEFHeader
from gdef_reporter.plotter_utils import plot_surface_to_axes


class GDEFSettings:
    """
    Stores all the settings used during measurement.

    :InstanceAttributes:
    bias_voltage
    calculated
    columns
    digital_loop
    direct_ac
    fft_type
    fixed_max
    fixed_min
    fixed_palette
    frequency_offset
    id
    invert_line_mean
    invert_plane_corr
    line_mean
    line_mean_order
    lines: total number of scan lines (including missing lines)
    loop_filter
    loop_gain
    loop_int
    max_height: scan area height [m]
    max_width: scan area width [m]
    measured_amplitude
    missing_lines: number of missing lines (e.g. due to aborted measurement)
    offset_pos
    offset_x
    offset_y
    phase_shift
    pixel_blend
    pixel_height: Pixel-height [m] (read-only property)
    pixel_width: Pixel-width [m] (read-only property)
    q_boost
    q_factor
    retrace
    retrace_type
    scan_direction
    scan_mode
    scan_speed: [µm/s]
    scanner_range
    set_point
    source_channel
    x_calib
    xy_linearized
    y_calib
    z_calib
    z_linearized
    z_unit
    zero_scan
    :EndInstanceAttributes:
    """
    def __init__(self):
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
        self._pixel_height = None

    @property
    def pixel_width(self) -> float:
        """Return pixel-width [m]."""
        return self._pixel_width  # self.max_width / self.columns

    @property
    def pixel_height(self) -> float:
        """Return pixel-height [m]."""
        return self._pixel_height  # self.max_height / self.lines

    def pixel_area(self) -> float:
        """Return pixel-area [m^2]"""
        return self.pixel_width * self._pixel_height

    def size_in_um_for_plot(self) -> Tuple[float, float, float, float]:
        """Returns the size of the scanned area as a tuple for use with matplotlib."""
        width_in_um = self.max_width * 1e6
        height_in_um = self.max_height * (self.lines - self.missing_lines) / self.lines * 1e6
        return 0.0, width_in_um, 0.0, height_in_um

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the scanned area (columns, lines). In case of aborted measurements, lines is reduced
        by the number of missing lines.
        """
        return self.columns - self.missing_lines, self.lines

    def _data_type_info(self) -> tuple:
        """
        Get information about the type of measurement data based on source_channel as a tuple.
        If data:_type is still missing in data_type_dict, the source_channel (int) is returned as unit!
        :return: tuple (data_type: str, unit: str, scaling_factor: float)
        """
        # todo: add more source_channel values to data_type_dict
        data_type_dict = {
            9: ("bending", "N", 1.0),
            11: ("topography", "nm", 1e9),
            12: ("phase", "deg", 1.0)  # factor 18.0 from gwyddion - seems to create too large values (e.g. 600 degree)
        }
        try:
            return data_type_dict[self.source_channel]
        except KeyError:
            print(f"source_channel {self.source_channel} not implemented in data_type_dict.")
            return ("SC", self.source_channel, 1.0)


class GDEFMeasurement:
    """
    Class containing data of a single measurement from \*.gdf file.

    :InstanceAttributes:
    comment: Comment text given for the measurement.
    filename:  Path of \*.pygdf file (only if loaded from pickled file, else None).
    gdf_basename: Path.stem of the imported \*.gdf file.
    gdf_block_id: Block ID in original \*.gdf file. Might be used to filter measurements.
    name: Returns a name of the measurement created from original \*.gdf filename and the gdf_block_id
    preview
    settings: GDEFSettings object
    values: Measurement values including corrections for background, offset etc.
    values_original: Original measurement data (read-only property)
    :EndInstanceAttributes:
    """
    def __init__(self):
        self._header: Optional[GDEFHeader] = None
        self._spm_image_file_version = None

        self.settings = GDEFSettings()

        self._values_original = None  # do not change! Use values instead
        self.values = None
        self.preview = None
        self.comment = ''

        self.gdf_basename = ""  # basename of original *.gdf file
        self.pygdf_filename: Optional[Path] = None  # basename of pickled *.pygdf
        self.gdf_block_id = None

        self.background_correction_type = None
        # self.background_corrected = False  # not implemented - might be better to save BGCorrectionType anyway

    @property
    def name(self) -> str:
        """Returns a name of the measurement created from original \*.gdf filename and the gdf_block_id"""
        if self.gdf_block_id is None:  # no measurement data loaded
            return ""
        return f"{self.gdf_basename}_block_{self.gdf_block_id:04}"

    @property
    def values_original(self) -> np.ndarray:
        """Returns a read-only np.ndarray with the original measurement data (before background correction etc.)."""
        return self._values_original

    def save_as_pickle(self, filename):
        """
        Save the measurement object using pickle. This is useful for example, if the corresponding
        \*.gdf file contains a lot of measurements, but only a few of them are needed. Take note, that pickle is not
        a save module to load data. Make sure to only use files from trustworthy sources.

        :param filename:
        :return: None
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file, 3)

    @staticmethod
    def load_from_pickle(filename: Path) -> "GDEFMeasurement":
        """
        Static method to load and return a measurement object using pickle. Take note, that pickle is not a save module
        to load data. Make sure to only use files from trustworthy sources.

        :param filename:
        :return: GDEFMeasurement
        """
        with open(filename, 'rb'):
            return pickle.load(filename)

    # todo: check possible types for filename (str, path, ...)
    def save_png(self, filename, max_figure_size=(4, 4), dpi: int = 300, transparent: bool = False):
        """
        Save a matplotlib.Figure of the measurement as a \*.png.
        :param filename:
        :param max_figure_size: Max size of the Figure. The final size might be smaller in x or y.
        :param dpi: (default 300)
        :param transparent: Set background transparent (default False).
        :return:
        """
        figure = self.create_plot(max_figure_size=max_figure_size, dpi=dpi)
        if figure:
            figure.savefig(filename, transparent=transparent, dpi=dpi)

    def set_topography_to_axes(self, ax: Axes, add_id: bool = False):
        """
        Sets the measurement data as diagram to a matplotlib Axes. If GDEFMeasurement.comment is not empty,
        the comment is used as title. Otherwise a default title with the type of measurement data is created.
        :param ax: Axes object to witch the topography is written.
        :param add_id: Adds block_id before title text (default False)
        :return: None
        """
        data_type, z_unit, z_factor = self.settings._data_type_info()
        if not isinstance(z_unit, str):
            z_unit = f"SC: {z_unit}"  # source_channel not in data_type_dict -> z_unit contains source_channel

        title = self.comment if self.comment else f"{data_type}"
        if add_id:
            title = f"{self.gdf_block_id}: {title}"

        if self.values is None:
            print(f"GDEFMeasurement {self.name} has values==None")
            return

        plot_surface_to_axes(ax, self.values, self.settings.pixel_width, title, z_unit, z_factor)

    def create_plot(self, max_figure_size=(4, 4), dpi=96, add_id: bool = False, trim: bool = True) -> Figure:
        """
        Returns a matplotlib figure of measurment data. If GDEFMeasurement.comment is not empty,
        the comment is used as title. Otherwise a default title with the type of measurement data is created.
        :param max_figure_size: Max. figure size. The actual figure size might be smaller.
        :param dpi: dpi value for Figure
        :param add_id:
        :param trim:
        :return: Figure
        """
        figure_max, ax = plt.subplots(figsize=max_figure_size, dpi=dpi)
        self.set_topography_to_axes(ax=ax, add_id=add_id)

        if not trim:
            return figure_max

        # remove unnecessary white areas (plot has a fixed aspect ratio due to scan area)
        tight_bbox = figure_max.get_tightbbox(figure_max.canvas.get_renderer())
        figure_tight, ax = plt.subplots(figsize=tight_bbox.size, dpi=dpi)
        self.set_topography_to_axes(ax=ax, add_id=add_id)
        figure_tight.tight_layout()
        return figure_tight

    def correct_background(self, correction_type: BGCorrectionType = BGCorrectionType.legendre_1, keep_offset: bool = False):
        """
        Corrects background using the given correction_type on values_original and save the result in values.
        If keep_z_offset is True, the mean value of dataset is preserved. Otherwise the average value is set to zero.
        Right now only changes topographical data. Also, the original data can be obtained again via
        GDEFMeasurement.values_original.

        :param correction_type: select type of background correction
        :param keep_offset: If True (default) keeps average offset, otherwise average offset is reduced to 0.
        :return: None
        """
        if not self.settings.source_channel == 11:  # only correct topography data
            return
        self.values = correct_background(self.values_original, correction_type=correction_type, keep_offset=keep_offset)
        self.background_correction_type = correction_type

    def get_summary_table_data(self) -> List[list]:  # todo: consider move method to utils.py
        """
        Create table data (list of list) summary of the measurement. The result can be used directly to fill a
        pptx-table with `python-ppxt-interface <https://github.com/natter1/python_pptx_interface/>`_.
        """
        result = [["source channel", self.settings.source_channel]]
        result.append(["retrace", self.settings.retrace])
        result.append(["missing lines", self.settings.missing_lines])
        result.append(["max width [m]", f"{self.settings.max_width:.2e}"])
        result.append(["max height [m]", f"{self.settings.max_height:.2e}"])
        result.append(["scan speed [µm/s]", f"{self.settings.scan_speed*1e6:.0f}"])
        if self.pygdf_filename:
            result.append(["basename", f"{self.pygdf_filename.stem}"])
        else:
            result.append(["name", f"{self.gdf_basename}_block_{self.gdf_block_id:03}"])

        return result
