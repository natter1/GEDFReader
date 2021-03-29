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

...