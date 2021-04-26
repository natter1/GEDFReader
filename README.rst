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
Module gdef_reader.gdef_importer
--------------------------------

class GDEFImporter
~~~~~~~~~~~~~~~~~~
This class is used to read data from a \*.gdf file (DME AFM) into python. This can be done like:

.. code:: python

    from gdef_reader.gdef_importer import GDEFImporter
    impported_data = GDEFImporter(gdf_path)  # gdf_path should be a pathlib.Path to a *.gdf file



**Methods:**

* **__init__**

    .. code:: python

        __init__(self, filename: Optional[pathlib.Path] = None)


    :filename: Path to \*.gdf file. If it is None (default), a file has to be loaded via GDEFImporter.load().

* **export_measurements**

    .. code:: python

        export_measurements(self, path: pathlib.Path = None, create_images: bool = False) -> List[gdef_reader.gdef_measurement.GDEFMeasurement]

    Create a list of GDEFMeasurement-Objects from imported data. The optional parameter create_images
    can be used to show a matplotlib Figure for each GDEFMeasurement (default value is False).

    :path: Save path for GDEFMeasurement-objects (and png's if create_images). No saved files, if None.

    :create_images: Show a matplotlib Figure for each GDEFMeasurement; used for debugging (default: False)

    :return: list of GDEFMeasurement-Objects

* **load**

    .. code:: python

        load(self, filename: Union[str, pathlib.Path]) -> None

    Import data from a \*.gdf file.

    :filename: Path to \*.gdf file.

    :return: None

**Instance Attributes:**

* basename: Path.stem of the imported \*.gdf file.
* bg_correction_type: BGCorrectionType for loaded measurements.
* keep_z_offset: If False (default), z-values for each imported measurement are corrected so that mean(z) == 0.

Module afm_tools.gdef_indent_analyzer
-------------------------------------

class GDEFIndentAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~
Class to analyze a GDEFMeasurment with an indent.



**Class Attributes:**

* max_pixel_radius_value
* pixel_radius_distance_matrix

**Methods:**

* **__init__**

    .. code:: python

        __init__(self, measurement: gdef_reader.gdef_measurement.GDEFMeasurement)


    :measurement: GDEFMeasurement with the indent to analyze.

* **add_map_with_indent_pile_up_mask_to_axes**

    .. code:: python

        add_map_with_indent_pile_up_mask_to_axes(self, ax: matplotlib.axes._axes.Axes, roughness_part=0.05) -> matplotlib.axes._axes.Axes

    Add a topography map with a color mask for pile-up to the given ax. Pile-up is determined as all pixels with
    z>0 + roughness_part \* z_max

    :ax: Axes object, to whitch the masked map should be added

    :roughness_part:

    :return: Axes

* **get_summary_table_data**

    .. code:: python

        get_summary_table_data(self) -> List[list]

    Returns a table (list of lists) with data of the indent. The result can be used directly to fill a pptx-table
    with `python-ppxt-interface <https://github.com/natter1/python_pptx_interface/>`_.

    :return:

Module gdef_reader.gdef_measurement
-----------------------------------

class GDEFMeasurement
~~~~~~~~~~~~~~~~~~~~~
Class containing data of a single measurement from \*.gdf file.



**Methods:**

* **__init__**

    .. code:: python

        __init__(self)

    Initialize self.  See help(type(self)) for accurate signature.

* **correct_background**

    .. code:: python

        correct_background(self, correction_type: afm_tools.background_correction.BGCorrectionType = <BGCorrectionType.legendre_1: 3>, keep_offset: bool = False)

    Corrects background using the given correction_type on values_original and save the result in values.
    If keep_z_offset is True, the mean value of dataset is preserved. Otherwise the average value is set to zero.
    Right now only changes topographical data. Also, the original data can be obtained again via
    GDEFMeasurement.values_original.


    :correction_type: select type of background correction

    :keep_offset: If True (default) keeps average offset, otherwise average offset is reduced to 0.

    :return: None

* **create_plot**

    .. code:: python

        create_plot(self, max_figure_size=(4, 4), dpi=96, add_id: bool = False, trim: bool = True) -> matplotlib.figure.Figure

    Returns a matplotlib figure of measurment data. If GDEFMeasurement.comment is not empty,
    the comment is used as title. Otherwise a default title with the type of measurement data is created.

    :max_figure_size: Max. figure size. The actual figure size might be smaller.

    :dpi: dpi value for Figure

    :add_id:

    :trim:

    :return: Figure

* **get_summary_table_data**

    .. code:: python

        get_summary_table_data(self) -> List[list]

    Create table data (list of list) summary of the measurement. The result can be used directly to fill a
    pptx-table with `python-ppxt-interface <https://github.com/natter1/python_pptx_interface/>`_.

* **load_from_pickle**

    .. code:: python

        load_from_pickle(filename: pathlib.Path) -> 'GDEFMeasurement'

    Static method to load and return a measurement object using pickle. Take note, that pickle is not a save module
    to load data. Make sure to only use files from trustworthy sources.


    :filename:

    :return: GDEFMeasurement

* **save_as_pickle**

    .. code:: python

        save_as_pickle(self, filename)

    Save the measurement object using pickle. This is useful for example, if the corresponding
    \*.gdf file contains a lot of measurements, but only a few of them are needed. Take note, that pickle is not
    a save module to load data. Make sure to only use files from trustworthy sources.


    :filename:

    :return: None

* **save_png**

    .. code:: python

        save_png(self, filename, max_figure_size=(4, 4), dpi: int = 300, transparent: bool = False)

    Save a matplotlib.Figure of the measurement as a \*.png.

    :filename:

    :max_figure_size: Max size of the Figure. The final size might be smaller in x or y.

    :dpi: (default 300)

    :transparent: Set background transparent (default False).

    :return:

* **set_topography_to_axes**

    .. code:: python

        set_topography_to_axes(self, ax: matplotlib.axes._axes.Axes, add_id: bool = False)

    Sets the measurement data as diagram to a matplotlib Axes. If GDEFMeasurement.comment is not empty,
    the comment is used as title. Otherwise a default title with the type of measurement data is created.

    :ax: Axes object to witch the topography is written.

    :add_id: Adds block_id before title text (default False)

    :return: None

**Instance Attributes:**

* background_correction_type
* comment: Comment text given for the measurement.
* gdf_basename: Path.stem of the imported \*.gdf file.
* gdf_block_id: Block ID in original \*.gdf file. Might be used to filter measurements.
* name: Returns a name of the measurement created from original \*.gdf filename and the gdf_block_id
* pixel_width
* preview
* pygdf_filename
* settings: GDEFSettings object
* values_original: Original measurement data (read-only property)
* values_original: Original measurement data (read-only property)

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
* line_mean_order
* line_mean_order
* lines: total number of scan lines (including missing lines)
* loop_filter
* loop_gain
* loop_int
* max_height: scan area height [m]
* max_width: scan area width [m]
* measured_amplitude
* missing_lines: number of missing lines (e.g. due to aborted measurement)
* offset_pos
* offset_x
* offset_y
* phase_shift
* pixel_blend
* pixel_height: Pixel-height [m] (read-only property)
* pixel_width: Pixel-width [m] (read-only property)
* q_boost
* q_factor
* retrace_type
* retrace_type
* scan_direction
* scan_mode
* scan_speed: [�m/s]
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

Module afm_tools.gdef_sticher
-----------------------------

class GDEFSticher
~~~~~~~~~~~~~~~~~
GDEFSticher combines/stiches several AFM area-measurements using cross-corelation to find the best fit.
To reduce calculation time, the best overlap position is only searched in a fraction of the measurement area
(defined by parameter initial_x_offset_fraction), and each measutrement is added to the right side.
Make sure the given list of measurements is ordered from left to right, otherwise wrong results are to be expected.
To evaluate the stiching, show_control_figures can be set to True. This creates a summary image
for each stiching step (using matplotlib plt.show()).



**Methods:**

* **__init__**

    .. code:: python

        __init__(self, measurements: List[gdef_reader.gdef_measurement.GDEFMeasurement], initial_x_offset_fraction: float = 0.35, show_control_figures: bool = False)


    :measurements:

    :initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching

    :show_control_figures:

* **stich**

    .. code:: python

        stich(self, initial_x_offset_fraction: float = 0.35, show_control_figures: bool = False) -> numpy.ndarray

    Stiches a list of GDEFMeasurement.values using cross-correlation.

    :initial_x_offset_fraction: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching

    :return: stiched np.ndarray

Module afm_tools.background_correction
--------------------------------------

class BGCorrectionType
~~~~~~~~~~~~~~~~~~~~~~
.. figure:: https://github.com/natter1/gdef_reader/raw/master/docs/images/BGCorrectionType_example01.png
    :width: 800pt

**Class Attributes:**

* gradient
* legendre_0
* legendre_1
* legendre_2
* legendre_3
* raw_data
