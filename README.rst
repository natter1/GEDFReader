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

**Functions:**

* **correct_background**

    .. code:: python

        correct_background(array2d: numpy.ndarray, correction_type: afm_tools.background_correction.BGCorrectionType, keep_offset: bool = False) -> Optional[numpy.ndarray]

    Returns a numpy.ndarray with corrections given by parameters. Input array2d is not changed.


    :array2d:

    :correction_type:

    :keep_offset:

    :return: ndarray

* **subtract_legendre_fit**

    .. code:: python

        subtract_legendre_fit(array2d: numpy.ndarray, keep_offset: bool = False, deg: int = 1) -> Optional[numpy.ndarray]

    Use a legendre polynomial fit of degree legendre_deg in X and Y direction to correct background.
    legendre_deg = 0 ... subtract mean value
    legendre_deg = 1 ... subtract mean plane
    legendre_deg = 2 ... subtract simple curved mean surface
    legendre_deg = 3 ... also corrects "s-shaped" distortion
    ...

* **subtract_mean_gradient_plane**

    .. code:: python

        subtract_mean_gradient_plane(array2d: numpy.ndarray, keep_offset: bool = False) -> Optional[numpy.ndarray]

    Returns 2d numpy.ndarray with subtracted mean gradient plane from given array2d. Using the gradient might give
     better results, when the measurement has asymmetric structures like large objects on a surface.
                                  _ _ _ _ _ _ _ _ _ _
    example: ____________________|                  |__

* **subtract_mean_level**

    .. code:: python

        subtract_mean_level(array2d: numpy.ndarray) -> numpy.ndarray

    Correct an offset in the array2d by subtracting the mean level.

    :array2d:

    :return: ndarray

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

Module gdef_reporter.plotter_utils
----------------------------------

**Functions:**

* **best_ratio_fit**

    .. code:: python

        best_ratio_fit(total_size: 'tuple[float, float]', single_size: 'tuple[float, float]', n: 'int') -> 'tuple[int, int]'

    Find best ratio of rows and cols to show n axes of ax_size on Figure with total_size.

    :total_size: total size available for n axes

    :single_size: size of one axes

    :n: number of axes to plot on total size

    :return: best ratio (rows and cols)

* **create_plot**

    .. code:: python

        create_plot(data_object: 'DataObject', pixel_width: 'float' = None, title: 'str' = '', max_figure_size: 'tuple[float, float]' = (4, 4), dpi: 'int' = 96, cropped: 'bool' = True) -> 'Figure'

    Creates a matplotlib Figure using given data_object. If cropped is True, the returned Figure has a smaller size
    than specified in max_figure_size.

    :data_object: DataObject with surface data

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :title: optional title (implemented as Figure suptitle)

    :max_figure_size: Max. figure size of returned Figure (actual size might be smaller if cropped).

    :dpi: dpi value of returned Figure

    :cropped: Crop the result Figure (default is True). Useful if aspect ratio of Figure and plot differ.

    :return: Figure

* **create_rms_plot**

    .. code:: python

        create_rms_plot(data_object_list: 'DataObjectList', pixel_width=None, label_list: 'Union[str, list[str]]' = None, title: 'str' = '', moving_average_n: 'int' = 200, x_offset=0, x_units: "Literal['�m', 'nm']" = '�m', subtract_average=True, plotter_style: 'PlotterStyle' = None) -> 'Figure'

    Creates a matplotlib figure, showing a graph of the root mean square roughness per column.

    :data_object_list: DataObjectList

    :pixel_width: has to be set, if data_object_list contains 1 or more np.ndarry (for varying values, use a list)

    :label_list: List with labels (str) for legend entries. If data_object_list is a dict, the keys are used.

    :title: Optional Figure title

    :moving_average_n: Number of columns to average over

    :x_offset: move data along x-axis

    :x_units: unit for x-axis (�m or nm)

    :subtract_average: Subtract average for each average_window (it might be better to subtract a global average)

    :plotter_style: PlotterStyle to format Figure-object (default: None -> use default format)

    :return: Figure

* **create_rms_with_error_plot**

    .. code:: python

        create_rms_with_error_plot(data_object_list: 'DataObjectList', pixel_width=None, label_list: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', average_n: 'int' = 8, x_units: "Literal['px', '�m', 'nm']" = '�m', y_units: "Literal['�m', 'nm']" = '�m', plotter_style: 'PlotterStyle' = None) -> 'Figure'

    Create a diagram, showing the root mean square roughness per column in for data in data_object_list.
    The error-bars are calculated as standard deviation of columns (average_n) used per data point.

    :data_object_list: DataObjectList

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :label_list: List with labels (str) for legend entries. If data_object_list is a dict, the keys are used.

    :title: Optional Figure title

    :average_n: Number of columns to average over

    :x_units: unit for x-axis (�m or nm)

    :y_units:

    :plotter_style: PlotterStyle to format Figure-object (default: None -> use default format)

    :return: None

* **create_summary_plot**

    .. code:: python

        create_summary_plot(data_object_list: 'DataObjectList', pixel_width: 'Optional[float]' = None, ax_title_list: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', figure_size: 'tuple[float, float]' = (16, 10), dpi: 'int' = 96) -> 'Figure'

    Creates a Figure with area-plots for each DataObject in data_object_list. Automatically determines best number of
    rows and cols. Works best, if all area-plots have the same aspect ratio.

    :data_object_list: DataObjectList

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :ax_title_list: Optional tiles for subplots

    :title: Figure title

    :figure_size:

    :dpi:

    :return: Figure

* **create_z_histogram_plot**

    .. code:: python

        create_z_histogram_plot(data_object_list: 'DataObjectList', pixel_width=None, labels: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', n_bins: 'int' = 200, units: "Literal['�m', 'nm']" = '�m', add_norm: 'bool' = False, plotter_style: 'PlotterStyle' = None) -> 'Figure'

    Also accepts a list of np.ndarray data (for plotting several histograms stacked)

    :data_object_list: DataObjectList

    :labels: labels for plotted data from values2d

    :title: Figure title; if empty, mu and sigma will be shown as axes subtitle(use title=None to prevent this)

    :n_bins: number of equally spaced bins for histogram

    :units: Can be set to �m or nm (default is �m).

    :add_norm: if True (default), show normal/gaussian probability density function for each distribution

    :plotter_style: PlotterStyle to format Figure-object (default: None -> use default format)

    :return: Figure

* **plot_rms_to_ax**

    .. code:: python

        plot_rms_to_ax(ax: 'Axes', data_object_list: 'DataObjectList', pixel_width=None, label_list: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', moving_average_n: 'int' = 200, x_offset=0, x_units: "Literal['�m', 'nm']" = '�m', subtract_average=True, plotter_style=None) -> 'None'

    Plot a diagram to ax, showing a the root mean square roughness per column in for data in data_object_list.

    :ax: Axes object to which the surface should be written

    :data_object_list: DataObjectList

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :label_list: List with labels (str) for legend entries. If data_object_list is a dict, the keys are used.

    :title: Optional axes title

    :moving_average_n: Number of columns to average over

    :x_offset: move data along x-axis

    :x_units: unit for x-axis (�m or nm)

    :subtract_average: Subtract average for each average_window (it might be better to subtract a global average)

    :plotter_style: PlotterStyle to format Figure-object (default: None -> use default format)

    :return: None

* **plot_rms_with_error_to_ax**

    .. code:: python

        plot_rms_with_error_to_ax(ax: 'Axes', data_object_list: 'DataObjectList', pixel_width=None, label_list: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', average_n: 'int' = 8, x_units: "Literal['px', '�m', 'nm']" = '�m', y_units: "Literal['�m', 'nm']" = '�m', plotter_style: 'PlotterStyle' = None)

    Plot a diagram to ax, showing the root mean square roughness per column in for data in data_object_list.
    The error-bars are calculated as standard deviation of columns (average_n) used per data point.

    :ax: Axes object to which the surface should be written

    :data_object_list: DataObjectList

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :label_list: List with labels (str) for legend entries. If data_object_list is a dict, the keys are used.

    :average_n: Number of columns to average over

    :x_units: unit for x-axis (�m or nm)

    :y_units:

    :plotter_style: PlotterStyle to format Figure-object (default: None -> use default format)

    :return: None

* **plot_to_ax**

    .. code:: python

        plot_to_ax(ax: 'Axes', data_object: 'DataObject', pixel_width: 'float' = None, title: 'str' = '', z_unit: "Literal['nm', '�m']" = 'nm') -> 'None'

    Plot values in data_object to given ax.

    :ax: Axes object to which the surface should be written

    :data_object: DataObject with surface data

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :title: Axes title (if '' -> shows mu and sigma (default); for no title set None)

    :z_unit: Units for z-Axis (color coded)

    :return: None

* **plot_z_histogram_to_ax**

    .. code:: python

        plot_z_histogram_to_ax(ax: 'Axes', data_object_list: 'DataObjectList', pixel_width: 'Optional[Union[float, list[float]]]' = None, label_list: 'Union[str, list[str]]' = None, title: 'Optional[str]' = '', n_bins: 'int' = 200, units: "Literal['�m', 'nm']" = '�m', add_norm: 'bool' = False, plotter_style=None) -> 'None'

    Also accepts a list of np.ndarray data (for plotting several histograms stacked)

    :ax: Axes object to which the surface should be written

    :data_object_list: DataObject or list[DataObject] with surface data

    :pixel_width: Pixel width/height in [m] (only used, if data_object has no pixel_width attribute)

    :label_list: labels for plotted data from values2d

    :title: Axes title; if empty, mu and sigma will be shown; to prevent any subtitle, set title=None

    :n_bins: number of equally spaced bins for histogram

    :units: Can be set to �m or nm (default is �m).

    :add_norm: if True (default), show normal/gaussian probability density function for each distribution

    :plotter_style: PlotterStyle to format Axes-object (default: None -> use default format)

    :return: None

* **save_figure**

    .. code:: python

        save_figure(figure: 'Figure', output_path: 'Path', filename: 'str', png: 'bool' = True, pdf: 'bool' = False) -> 'None'

    Helper function to save a matplotlib figure as png and or pdf. Automatically creates output_path, if necessary.
    Does nothing if given output_path is None.

* **split_dict_in_data_and_label_list**

    .. code:: python

        split_dict_in_data_and_label_list(data_dict_list: 'dict[str:DataObject]')

    deprecated
