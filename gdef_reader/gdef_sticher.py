from pathlib import Path
from scipy import signal
from scipy import misc
import numpy as np

from gdef_reader.gdef_importer import GDEFImporter
from gdef_reader.gdef_importer import GDEFMeasurement
from gdef_reader.utils import load_pygdf_measurements

import matplotlib.pyplot as plt
import png

samplename = "cantilever03"
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


# get rms roughness
def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

def create_absolute_gradient_array(array2d, cutoff = 1.0):
    result = np.gradient(array2d)  # [0]
    result = np.sqrt(result[0] ** 2 + result[1] ** 2)
    max_grad = np.nanmax(result)
    with np.nditer(result, op_flags=['readwrite']) as it:
        for x in it:
            if x is not np.nan and x > cutoff * max_grad:
                x[...] = 0 #np.nan
    return result

def create_image(array2d):
    data_min = np.nanmin(array2d)
    array2d = (array2d - min(0, data_min)) / (np.nanmax(array2d) - min(0, data_min))  # normalize the data to 0 - 1
    array2d = 255 * array2d  # Now scale by 255
    return array2d.astype(np.uint8)

x_pos = []
y_rms = []
averaging = 10
gradient_cutoff = 0.1


for i in range(data_stiched.shape[1]-averaging):
    x_pos.append((i+averaging/2.0)*measurements[0].settings.pixel_width*1e6)
    y_rms.append(nanrms(data_stiched[:, i:i+averaging]))
    # y_rms.append(np.nanmean(gradient_stiched_data[:, i:i+averaging]))

fig, (ax_rms) = plt.subplots(1, 1, figsize=(10, 6))
ax_rms.plot(x_pos, y_rms, 'r')
ax_rms.set_xlabel("[µm]")
ax_rms.set_ylabel(f"rms (averaged over {averaging} line(s))")
fig.show()


#data_image = np.vstack(map(np.uint16, data_stiched))
#png.from_array(data_image, mode="L").save("delme_stich.png")

# img = create_image(data_stiched)
# png.from_array(img, mode="L").save(f"{samplename}_stiched.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 1.0)),
#                mode="L").save(f"{samplename}_gradient_cutoff_1-0.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.9)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-9.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.8)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-8.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.7)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-7.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.6)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-6.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.5)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-5.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.4)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-4.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.3)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-3.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.2)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-2.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.1)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-1.png")
#
# png.from_array(create_image(create_absolute_gradient_array(data_stiched, 0.05)),
#                mode="L").save(f"{samplename}_gradient_cutoff_0-05.png")



# fig, (ax_100, ax_090, ax_080, ax_070, ax_060, ax_050, ax_040, ax_030, ax_020, ax_015, ax_010, ax_005) \
#     = plt.subplots(12, 1, figsize=(12, 20))
#
# ax_100.imshow(create_absolute_gradient_array(data_stiched, 1.0), cmap='gray')
# ax_100.set_title('gradient cutoff 1.0')
# ax_100.set_axis_off()


cutoff_percent_list = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 12, 10, 8, 5]
cutoff_percent_list = [100, 60, 20, 15, 12, 10, 8, 5, 3, 2, 1]
averaging = 5

fig, ax_list = plt.subplots(len(cutoff_percent_list), 1, figsize=(len(cutoff_percent_list)*0.4, 13))
figure_gradient_rms, (ax_gradient_rms) = plt.subplots(1, 1, figsize=(10, 10))
ax_gradient_rms.set_xlabel("[µm]")
ax_gradient_rms.set_ylabel(f"absolute gradient rms (averaged over {averaging} line(s))")

for i, percent in enumerate(cutoff_percent_list):
    absolut_gradient_array = create_absolute_gradient_array(data_stiched, percent/100.0)
    ax_list[i].imshow(absolut_gradient_array, cmap='gray')
    ax_list[i].set_title(f'gradient cutoff {percent}%')
    ax_list[i].set_axis_off()

    x_pos = []
    y_gradient_rms = []
    for i in range(absolut_gradient_array.shape[1] - averaging):
        x_pos.append((i + averaging / 2.0) * measurements[0].settings.pixel_width * 1e6)
        y_gradient_rms.append(nanrms(absolut_gradient_array[:, i:i + averaging]))
    ax_gradient_rms.plot(x_pos, y_gradient_rms, label=f"{percent}%")

fig.show()
ax_gradient_rms.legend()
figure_gradient_rms.show()
