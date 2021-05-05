"""
@author: Nathanael Jöhrmann
"""
from pathlib import Path, PurePath
from typing import List, Union

from afm_tools.background_correction import BGCorrectionType
from gdef_reporter.gdef_reporter import GDEFContainerList, GDEFContainer, GDEFReporter


def create_gdef_reporter(gdf_paths: Union[list[Path], Path],
                         filter_dict: dict = None,
                         bg_correction_type: BGCorrectionType = BGCorrectionType.legendre_1,
                         keep_offset: bool = False)\
        -> GDEFReporter:
    """
    Creates a GDEFReporter with all the data found at gdef_paths (list of files and/or folders as pathlib.Path).
    Using filter_dict, it is possible to define data that should be ignored (the data is still loaded and can be used
    explicitly).
    :param gdf_paths: (optional: list of) Pathlib Path object(s) to *.gdf file or folder (load all *.gdf files in there)
    :param filter_dict: dict with filename as kay and a list of IDs as value, used to set gdf_container_list filter
    :param bg_correction_type: define a type of background correction, that is applied to all imported measurements
    :param keep_offset: defines if z-offset is kept, or if all measurements where set to avg. z = 0 (default)
    :return: GDEFReporter
    """
    gdf_container_list = GDEFContainerList()
    if isinstance(gdf_paths, PurePath):
        gdf_paths = [gdf_paths]

    for gdf_path in gdf_paths:
        if gdf_path.is_file():
            gdf_container_list.append(GDEFContainer(gdf_path))
        else:
            for gdf_file in gdf_path.glob("*.gdf"):
                gdf_container_list.append(GDEFContainer(gdf_file))
    gdf_container_list.correct_backgrounds(bg_correction_type, keep_offset)
    gdf_container_list.set_filter_ids(filter_dict)

    return GDEFReporter(gdf_container_list)


