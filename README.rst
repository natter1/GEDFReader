GDEFReader
==========
.. image:: https://img.shields.io/pypi/v/gdef_reader.svg
    :target: https://pypi.org/project/gdef_reader/

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

Content
-------
  * `GDEFImporter <#class-gdefimporter>`__: Class to import \*.gdf files
  * `Single measurements <single-measurements>`__
     + `class GDEFMeasurement <#class-gdefmeasurement>`__: Class containing data of a single measurement from \*.gdf file.
     + `class GDEFSettings <#class-gdefsettings>`__: Class containing all settings of a measurement.
  * `class GDEFSticher <#class-gdefsticher>`__: Tool to stich several measurements together using cross correlation.
  * `class GDEFReporter <#class-gdefreporter>`__: Tool to create reports (\*.pptx, \*.png(?), matplotlib figures(?))
     + `class GDEFContainer <#class-gdefcontainer>`__: Helper class for measurement filtering and background correction.
     + `class GDEFContainerList <#class-GDEFContainerList>`__
  * `utils.py <#utilspy>`__: A collection of useful functions, eg. to generate PDF or PNG from \*.pptx (needs PowerPoint installed)
  * `Examples <#example>`__: Collection of examples demonstrating how to use gdef_reader.


class GDEFImporter
------------------

This class is used to read data from a \*.gdf file (DME AFM) into python. This can be done like:

.. code:: python

    from gdef_reader.gdef_importer import GDEFImporter
    impported_data = GDEFImporter(gdf_path)  # gdf_path should be a pathlib.Path to a *.gdf file


**Methods defined:**

* export_measurements
    Create a list of GDEFMeasurement-Objects from imported data. The optional parameter create_images can be used to
    show a matplotlib Figure for each GDEFMeasurement (default value is False).
* load
    Import data from a \*.gdf file.

**Properties defined:**

* **basename**: Path.stem of the imported \*.gdf file.


Single GDEFMeasurement
-------------------
A \*.gdf file can contain many AFM measurements. To handle a single measurement the class GDEFMeasurement is used.
All the settings used during that specific measurement are stored in a GDEFSettings object.


class GDEFMeasurement
~~~~~~~~~~~~~~~~~~~~~
Class containing data of a single measurement from \*.gdf file.

**Methods defined:**

* export_measurements
    Create a list of GDEFMeasurement-Objects from imported data. The optional parameter create_images can be used to
    show a matplotlib Figure for each GDEFMeasurement (default value is False).
* load
    Import data from a \*.gdf file.

**Properties defined:**

* **basename**: Path.stem of the imported \*.gdf file.