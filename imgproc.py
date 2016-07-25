import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import isfile, join, isdir
import os
from os import listdir
from scipy import misc

__ending__ = "_cropped.bmp"  # ending of filename for files to be processed

parser = argparse.ArgumentParser()
parser.add_argument('path', help="Path to directory with images")
parser.add_argument('cal', help="mm/pixel in images", type=float)
args = parser.parse_args()
cal = args.cal
path = args.path

files = [f for f in listdir(path) if (isfile(join(path, f)) and f.__contains__(__ending__))]

data_path = join(path, "data/")

# only create new directory if there are files to process
if len(files) > 0:
    if not isdir(data_path):
        os.makedirs(data_path)

output = open(join(data_path, "radii.csv"), mode='w')
output.write("File name,2rms X radius (mm),2rms Y radius (mm)\n")

for f in files:
    print "Processing " + f
    img = misc.imread(join(path, f))
    max_val = np.amax(img)
    min_val = np.amin(img)
    img -= min_val  # filter out background

    img_sum = np.sum(img)
    i_avg = 0  # x centroid
    j_avg = 0  # y centroid
    for (j, i), v in np.ndenumerate(img):
        i_avg += (i * v)  # x moment
        j_avg += (j * v)  # y moment
    i_avg /= img_sum
    j_avg /= img_sum

    # plot intensity distributions along axes that go through the beam centroid (should be K-V-like)
    p = round(i_avg)
    q = round(j_avg)
    y_cent_row = img[:, q]
    x_cent_col = img[p, :]
    plt.figure(1)
    plt.subplot(121)
    plt.plot((range(y_cent_row.shape[0]) - j_avg) * cal,
             y_cent_row / float(max_val))  # distance from x centroid vs. relative intensity
    plt.subplot(122)
    plt.plot((range(x_cent_col.shape[0]) - i_avg) * cal, x_cent_col / float(max_val))
    # plt.savefig(join(data_path, f[:-4] + "_plots.png"))  # the -4 removes the .bmp extension
    plt.show()
    # plt.close()

    i_ssr = 0
    j_ssr = 0
    for (j, i), v in np.ndenumerate(img):
        i_ssr += (i - i_avg) ** 2 * v
        j_ssr += (j - j_avg) ** 2 * v
    x_rms = np.sqrt(i_ssr) / img_sum * cal
    y_rms = np.sqrt(j_ssr) / img_sum * cal
    output.write("{0},{1},{2}\n".format(f, 2 * x_rms, 2 * y_rms))

if len(files) > 0:
    output.close()