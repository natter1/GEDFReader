import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_measurement import GDEFMeasurement


class GDEFContainer:
    """
    Container class for all measurements inside a *.gdf-file
    """
    def __init__(self, gdf_path: Path):
        self.basename: str = gdf_path.stem
        self.base_path_name = gdf_path.parent.stem
        self.path: Optional[Path] = gdf_path
        self.last_modification_datetime: datetime = datetime().fromtimestamp(os.path.getmtime(gdf_path))
        self.measurements: List[GDEFMeasurement] = GDEFImporter(gdf_path).export_measurements()
        self.filter_ids: List[int] = []

    # def set_filter_ids(self, ids: List[int]):
    #     self.filter_ids = ids

    @property
    def filtered_measurements(self) -> List[GDEFMeasurement]:
        return [x for x in self.measurements if not x.gdf_block_id in self.filter_ids]

    def save(self, path: Path):
        if path:
            path.mkdir(parents=True, exist_ok=True)
            filename = path.joinpath(f"{self.basename}.pygdf")
        with open(filename, 'wb') as file:
            pickle.dump(self, file, 3)


class GDEFReporter:
    def __init__(self, gdf_containers: Optional[GDEFContainer, List[GDEFContainer]] = None):
        if isinstance(gdf_containers, GDEFContainer):
            self.gdf_containers = [gdf_containers]
        else:
            self.gdf_containers: List[GDEFContainer] = gdf_containers

    def dummy(self):
        pass



# def creation_date(path_to_file):
#     """
#     Try to get the date that a file was created, falling back to when it was
#     last modified if that isn't possible.
#     See http://stackoverflow.com/a/39501288/1709587 for explanation.
#     """
#     if platform.system() == 'Windows':
#         return time.ctime(os.path.getctime(path_to_file))
#     else:
#         stat = os.stat(path_to_file)
#         try:
#             return time.ctime(stat.st_birthtime)
#         except AttributeError:
#             # We're probably on Linux. No easy way to get creation dates here,
#             # so we'll settle for when its content was last modified.
#             return time.ctime(stat.st_mtime)