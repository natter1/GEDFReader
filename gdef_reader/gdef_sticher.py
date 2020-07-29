from pathlib import Path
from scipy import signal
from scipy import misc
import numpy as np

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_importer import GDEFMeasurement
from gdef_reader.utils import load_pygdf_measurements

import matplotlib.pyplot as plt
import png

folder = '..\\delme'
path = Path.cwd().parent.joinpath("delme")
measurements = load_pygdf_measurements(path)
print(measurements)

data01 = measurements[0].values
data02 = measurements[1].values
data03 = measurements[2].values
data04 = measurements[3].values
data05 = measurements[4].values
data06 = measurements[5].values

#data02 = data02[0:, 15:]

def stich(data01, data02, data01_x_offset):
    """

    :param data01:
    :param data02:
    :param data01_x_offset: used to specify max. overlap area, thus increasing speed and reducing risk of wrong stiching
    :return: np.2darray
    """
    data02_x_offset_right = data01.shape[1] - data01_x_offset
    correlation = signal.correlate2d(data01[:, data01_x_offset:],
                                     data02[:, :data02_x_offset_right])  # , boundary="wrap")  # using "wrap" ensures, that x, y below can be used directly

    reduced_correlation = correlation[:, data02_x_offset_right:]  # make sure, data02 is appended on right side
                                                                  # this reduces risk of wrong stiching, but measurements have to be in right order

    y, x = np.unravel_index(np.nanargmax(reduced_correlation), reduced_correlation.shape)  # find (first) best match
    y, x = y - data02.shape[0] + 1, x + 1 + data01_x_offset  # - data02_x_offset_right)  # test with two identical datasets -> should give: y, x = 0, 0

    fig, (ax_orig, ax_template, ax_corr, ax_stich) = plt.subplots(4, 1, figsize=(6, 20))

    ax_orig.imshow(data01, cmap='gray')
    ax_orig.set_title('data01')
    ax_orig.set_axis_off()

    ax_template.imshow(data02, cmap='gray')
    ax_template.set_title('data02')
    ax_template.set_axis_off()

    ax_corr.imshow(correlation, cmap='gray')
    ax_corr.set_title('Cross-correlation')
    ax_corr.set_axis_off()

    ax_orig.plot(x, y, 'ro')

    data01_x0 = - min(0, x)
    data01_y0 = - min(0, y)
    data02_x0 = max(0, x)
    data02_y0 = max(0, y)

    data01_height = data01.shape[0] + data01_y0
    data01_width = data01.shape[1] + data01_x0

    data02_height = data02.shape[0] + data02_y0
    data02_width = data02.shape[1] + data02_x0

    data_stiched = np.full([max(data01_height, data02_height), max(data01_width, data02_width)], np.nan)

    data_stiched[data01_y0:data01_height, data01_x0:data01_width] = data01
    data_stiched[data02_y0:data02_height, data02_x0:data02_width] = data02

    ax_stich.set_title('stiched')
    ax_stich.set_axis_off()
    ax_stich.imshow(data_stiched, cmap='gray')

    fig.show()

    return data_stiched


data01_x_offset_right = round(data01.shape[1] * 0.35)
data_stiched = stich(data01, data02, data01.shape[1]-data01_x_offset_right)
data_stiched = stich(data_stiched, data03, data_stiched.shape[1]-data01_x_offset_right)
data_stiched = stich(data_stiched, data04, data_stiched.shape[1]-data01_x_offset_right)
data_stiched = stich(data_stiched, data05, data_stiched.shape[1]-data01_x_offset_right)
data_stiched = stich(data_stiched, data06, data_stiched.shape[1]-data01_x_offset_right)

data=data_stiched
data_min = np.nanmin(data)
data_max = np.nanmax(data)
data = (data - min(0,data_min)) / (np.nanmax(data)- min(0,data_min)) # normalize the data to 0 - 1
data = 255 * data # Now scale by 255
img = data.astype(np.uint8)


#data_image = np.vstack(map(np.uint16, data_stiched))
#png.from_array(data_image, mode="L").save("delme_stich.png")
png.from_array(img, mode="L").save("delme_stich.png")