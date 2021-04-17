"""
This file contains tests for gdef_importer.py.
@author: Nathanael Jöhrmann
"""
# todo: add temporary folder and test export of *.pygdf and *.png (export_measurements())
from pathlib import Path

import pytest

from gdef_reader.gdef_importer import GDEFImporter

example_01_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")


@pytest.fixture(scope='session')
def gdef_importer():
    importer = GDEFImporter(example_01_path)
    yield importer


class TestGDEFImporter:
    def test_init(self):
        importer = GDEFImporter(example_01_path)
        assert importer.basename == "example_01"

    def test_export_measurements(self, gdef_importer):
        measurements = gdef_importer.export_measurements()
        assert len(measurements) == 4

    def test_export_measurements_with_create_images(self, gdef_importer):
        measurements = gdef_importer.export_measurements(create_images=True)
        assert len(measurements) == 4

    def test_load(self):
        importer = GDEFImporter()
        importer.load(example_01_path)
        assert importer.basename == "example_01"
