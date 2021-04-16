from pathlib import Path

import matplotlib.pyplot as plt

from afm_tools.background_correction import BGCorrectionType
from gdef_reader.gdef_importer import GDEFImporter

gdf_path = Path.cwd().parent.joinpath("resources").joinpath("example_01.gdf")
docs_img_path = Path.cwd().parent.joinpath("docs").joinpath("images")
gdf_importer = GDEFImporter(gdf_path)

measurements = gdf_importer.export_measurements()
measurement = measurements[0]
measurement.comment = "imported (default settings)"

# fig = measurement.create_plot()
# fig.show()
#
# for correction_type in BGCorrectionType:
#     measurement.correct_background(correction_type)
#     measurement.comment = correction_type.name
#     fig = measurement.create_plot()
#     fig.show()

correction_summary_fig, axes = plt.subplots(3, 2, dpi=150, figsize=(8, 4.5))

ax_list = axes.flatten()

for i, correction_type in enumerate(BGCorrectionType):
    measurement.correct_background(correction_type,keep_offset=True)
    measurement.comment = correction_type.name
    measurement.set_topography_to_axes(ax_list[i], add_id=False)

for ax in ax_list[len(BGCorrectionType):]:
    ax.remove()
correction_summary_fig.tight_layout()
correction_summary_fig.show()
# correction_summary_fig.savefig(docs_img_path.joinpath("BGCorrectionType_example01.png"))