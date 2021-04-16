GDEFReader
==========
.. image:: https://img.shields.io/pypi/v/GDEFReader.svg
    :target: https://pypi.org/project/GDEFReader/

.. image:: http://img.shields.io/:license-MIT-blue.svg?style=flat-square
    :target: http://badges.MIT-license.org

|

.. figure:: https://github.com/natter1/gdef_reader/raw/master/docs/images/example_overview_image.png
    :width: 800pt

|


Tool to read \*.gdf files (DME AFM)

Features
--------

* import measurements from \*.gdf file into python
* create maps using matplotlib
* analyze nanoindents
* stich measurements
* create customizable output (e.g. \*.png or power point presentations)


.. contents:: Table of Contents

API documentation
=================
Module gdef_reader.gdef_measurement
-----------------------------------

class GDEFMeasurement
~~~~~~~~~~~~~~~~~~~~~
Class containing data of a single measurement from \*.gdf file.

:InstanceAttributes:
gdf_basename: Path.stem of the imported \*.gdf file.
:EndInstanceAttributes:

**Methods:**

* **__init__**

    .. code:: python

        __init__(self)

    Initialize self.  See help(type(self)) for accurate signature.

* **correct_background**

    .. code:: python

        correct_background(self, correction_type: afm_tools.background_correction.BGCorrectionType = <BGCorrectionType.legendre_1: 3>, keep_offset: bool = False)

    Corrects background using the given correction_type on values_original and save the result in values.
    If keep_offset is True, the mean value of dataset is preserved. Otherwise the average value is set to zero.
    Right now only changes topographical data. Also, the original data can be obtained again via
    GDEFMeasurement.values_original.


    :correction_type: select type of background correction

    :keep_offset: If True (default) keeps average offset, otherwise average offset is reduced to 0.

    :return: None

* **create_plot**

    .. code:: python

        create_plot(self, max_figure_size=(4, 4), dpi=96) -> Union[matplotlib.figure.Figure, NoneType]


* **get_summary_table_data**

    .. code:: python

        get_summary_table_data(self) -> List[list]

    Create table data (list of list) summary of the measurement. The result can be used directly to fill a
    pptx-table with `python-ppxt-interface <https://github.com/natter1/python_pptx_interface/>`_.

* **load**

    .. code:: python

        load(filename: pathlib.Path) -> 'GDEFMeasurement'

    Load a measurement object using pickle. Take note, that pickle is not a save module to load data.
    Make sure to only use files from trustworthy sources.


    :filename:

    :return:

* **save**

    .. code:: python

        save(self, filename)

    Save the measurement object using pickle. This is useful for example, if the corresponding
    \*.gdf file contains a lot of measurements, but only a few of them are needed. Take note, that pickle is not
    a save module to load data. Make sure to only use files from trustworthy sources.


    :filename:

    :return:

* **save_png**

    .. code:: python

        save_png(self, filename, max_figure_size=(4, 4), dpi: int = 300, transparent: bool = False)

    Save a matplotlib.Figure aof the measurement as a \*.png.

    :filename:

    :max_figure_size: Max size of the Figure. The final size might be smaller in x or y.

    :dpi: (default 300)

    :transparent: Set background transparent (default False).

    :return:

* **set_topography_to_axes**

    .. code:: python

        set_topography_to_axes(self, ax: matplotlib.axes._axes.Axes)


**Instance Attributes:**

* background_corrected
* comment
* filename
* gdf_basename('', 'gdf_basename', ': Path.stem of the imported \\*.gdf file.')
* gdf_block_id
* name
* preview
* settings
* values
* values_original

class GDEFSettings
~~~~~~~~~~~~~~~~~~
Stores all the settings used during measurement.

**Methods:**

* **__init__**

    .. code:: python

        __init__(self)

    Initialize self.  See help(type(self)) for accurate signature.

* **pixel_area**

    .. code:: python

        pixel_area(self) -> float

    Return pixel-area [m^2]

* **shape**

    .. code:: python

        shape(self) -> Tuple[int, int]

    Returns the shape of the scanned area (columns, lines). In case of aborted measurements, lines is reduced
    by the number of missing lines.

* **size_in_um_for_plot**

    .. code:: python

        size_in_um_for_plot(self) -> Tuple[float, float, float, float]

    Returns the size of the scanned area as a tuple for use with matplotlib.

**Instance Attributes:**

* aux_gain
* bias_voltage
* calculated
* columns
* digital_loop
* direct_ac
* fft_type
* fixed_max
* fixed_min
* fixed_palette
* frequency_offset
* id
* invert_line_mean
* invert_plane_corr
* line_mean
* line_mean_order
* lines
* loop_filter
* loop_gain
* loop_int
* max_height
* max_width
* measured_amplitude
* missing_lines
* offset_pos
* offset_x
* offset_y
* phase_shift
* pixel_blend
* pixel_height
* pixel_width
* q_boost
* q_factor
* retrace
* retrace_type
* scan_direction
* scan_mode
* scan_speed
* scanner_range
* set_point
* source_channel
* x_calib
* xy_linearized
* y_calib
* z_calib
* z_linearized
* z_unit
* zero_scan

Module afm_tools.background_correction
--------------------------------------

class BGCorrectionType
~~~~~~~~~~~~~~~~~~~~~~
An enumeration.

**Class Attributes:**

* gradient
* legendre_0
* legendre_1
* legendre_2
* legendre_3
* raw_data
