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
* create customizable output (e.g. \*.png or power point presentstions)


.. contents:: Table of Contents

API documentation
=================
Module gdef_reader.gdef_importer
--------------------------------

class GDEFImporter
~~~~~~~~~~~~~~~~~~
This class is used to read data from a \*.gdf file (DME AFM) into python. This can be done like:

.. code:: python

    from gdef_reader.gdef_importer import GDEFImporter
    impported_data = GDEFImporter(gdf_path)  # gdf_path should be a pathlib.Path to a *.gdf file

Attributes:

    * basename: Path.stem of the imported \*.gdf file.

**Methods:**

* **export_measurements**

    .. code:: python

        export_measurements(self, path: pathlib.Path = None, create_images: bool = False) -> List[gdef_reader.gdef_measurement.GDEFMeasurement]

    Create a list of GDEFMeasurement-Objects from imported data. The optional parameter create_images
    can be used to show a matplotlib Figure for each GDEFMeasurement (default value is False).
    :param path: Save path for GDEFMeasurement-objects. No saved files, if None.
    :param create_images:
    :return: list of GDEFMeasurement-Objects

* **load**

    .. code:: python

        load(self, filename: Union[str, pathlib.Path])

    Import data from a \*.gdf file.
    :param filename:
    :return:

**Instance Variables:**

* base_blocks
* basename
* blocks
* buffer
* header

Module gdef_reader.gdef_indent_analyzer
---------------------------------------

class GDEFIndentAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~

**Methods:**

* **add_indent_pile_up_mask_to_axes**

    .. code:: python

        add_indent_pile_up_mask_to_axes(self, ax: matplotlib.axes._axes.Axes, roughness_part=0.05) -> matplotlib.axes._axes.Axes


* **get_summary_table_data**

    .. code:: python

        get_summary_table_data(self)


**Instance Variables:**


Module gdef_reader.gdef_measurement
-----------------------------------

class GDEFMeasurement
~~~~~~~~~~~~~~~~~~~~~
Class containing data of a single measurement from \*.gdf file.

Attributes:

    * basename: Path.stem of the imported \*.gdf file.

**Methods:**

* **correct_background**

    .. code:: python

        correct_background(self, use_gradient_plane: bool = True, legendre_deg: int = 1, keep_offset: bool = False)

    Subtract legendre polynomial fit of degree legendre_deg from values_original and save the result in values.
    If keep_offset is true, the mean value of dataset is preserved. Right now only changes topographical data.
    average value to zero and subtract tilted background-plane.

* **create_plot**

    .. code:: python

        create_plot(self, max_figure_size=(4, 4), dpi=96) -> Union[matplotlib.figure.Figure, NoneType]


* **get_summary_table_data**

    .. code:: python

        get_summary_table_data(self)


* **load**

    .. code:: python

        load(filename) -> 'GDEFMeasurement'


* **save**

    .. code:: python

        save(self, filename)


* **save_png**

    .. code:: python

        save_png(self, filename, max_figure_size=(4, 4), dpi: int = 300, transparent: bool = False)


* **set_topography_to_axes**

    .. code:: python

        set_topography_to_axes(self, ax: matplotlib.axes._axes.Axes)


**Instance Variables:**

* background_corrected
* comment
* filename
* gdf_basename
* gdf_block_id
* name
* preview
* settings
* values
* values_original

class GDEFSettings
~~~~~~~~~~~~~~~~~~

**Methods:**

* **pixel_area**

    .. code:: python

        pixel_area(self) -> float


* **shape**

    .. code:: python

        shape(self) -> Tuple[int, int]


* **size_in_um_for_plot**

    .. code:: python

        size_in_um_for_plot(self) -> Tuple[float, float, float, float]


**Instance Variables:**

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

Module gdef_reader.gdef_sticher
-------------------------------

class GDEFSticher
~~~~~~~~~~~~~~~~~

**Methods:**

* **stich**

    .. code:: python

        stich(self, initial_x_offset_fraction: float = 0.35, show_control_figures: bool = False) -> numpy.ndarray

    Stiches a list of GDEFMeasurement.values using cross-correlation.
    :param initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
    :return: stiched np.ndarray

**Instance Variables:**

