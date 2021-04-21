"""
@author: Nathanael Jöhrmann
"""
import numpy as np

from afm_tools.gdef_sticher import GDEFSticher
from gdef_reporter.gdef_reporter import GDEFContainerList


def create_gdef_sticher_dict(gdf_containers: GDEFContainerList, reverse_flag_dict: dict, initial_x_offset_fraction,
                             show_control_figures=False, filter_below_to_nan_value = None):
    result = {}
    for gdf_container in gdf_containers:
        measurements = gdf_container.filtered_measurements
        if reverse_flag_dict[gdf_container.basename]:
            measurements.reverse()
        sticher = GDEFSticher(measurements, initial_x_offset_fraction, show_control_figures=show_control_figures)
        if filter_below_to_nan_value is not None:
            sticher.stiched_data[sticher.stiched_data < filter_below_to_nan_value] = np.nan
        result[gdf_container.basename] = sticher
    return result