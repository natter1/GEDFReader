"""
@author: Nathanael Jöhrmann
"""
from pathlib import Path, PurePath
from typing import List, Union

from gdef_reporter.gdef_reporter import GDEFContainerList, GDEFContainer, GDEFReporter


def create_gdef_reporter(gdf_paths: Union[List[Path], Path], filter_dict: dict = None, use_gradient_plane: bool = False,
                         legendre_deg: int = 1, keep_offset: bool = False) -> GDEFReporter:
    gdf_container_list = GDEFContainerList()
    if isinstance(gdf_paths, PurePath):
        gdf_paths = [gdf_paths]

    for gdf_path in gdf_paths:
        if gdf_path.is_file():
            gdf_container_list.append(GDEFContainer(gdf_path))
        else:
            for gdf_file in gdf_path.glob("*.gdf"):
                gdf_container_list.append(GDEFContainer(gdf_file))
    gdf_container_list.correct_backgrounds(use_gradient_plane, legendre_deg, keep_offset)
    gdf_container_list.set_filter_ids(filter_dict)

    return GDEFReporter(gdf_container_list)
