"""
This file contains tests for gdef_importer.py.
@author: Nathanael Jöhrmann
"""
# todo: add temporary folder and test export of *.pygdf and *.png (export_measurements())

from gdef_reader.gdef_importer import GDEFImporter
from tests.conftest import AUTO_SHOW


class TestGDEFImporter:
    def test_init(self, gdf_example_01_path):
        importer = GDEFImporter(gdf_example_01_path)
        assert importer.basename == "example_01"

    def test_export_measurements(self, gdef_importer):
        measurements = gdef_importer.export_measurements()
        assert len(measurements) == 4

    def test_export_measurements_with_create_images(self, gdef_importer):
        measurements = gdef_importer.export_measurements(create_images=AUTO_SHOW)
        assert len(measurements) == 4

    def test_load(self, gdf_example_01_path):
        importer = GDEFImporter()
        importer.load(gdf_example_01_path)
        assert importer.basename == "example_01"
