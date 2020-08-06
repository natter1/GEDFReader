import glob
# import os

from pathlib import Path

import pickle
from typing import List

from gdef_reader.gdef_measurement import GDEFMeasurement
from gdef_reader.utils import load_pygdf_measurements, create_pptx_for_nanoindents, create_png_for_nanoindents
from gdef_reader.etit169_pptx_template import TemplateETIT169


from pptx_tools.creator import PPTXCreator, PPTXPosition

def main():

    # path = Path.cwd().parent.joinpath("output").joinpath("Al_1µm_s001").joinpath("best")
    path = Path.cwd().parent.joinpath("output").joinpath("Al_1µm_s001")
    pptx_filename = "Al_1µm_s001"

    create_pptx_for_nanoindents(path, pptx_filename, pptx_template=TemplateETIT169())
    # create_png_for_nanoindents(path, path.joinpath("delmedelme"))

if __name__ == '__main__':
    main()