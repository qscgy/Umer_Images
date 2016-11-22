import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import isfile, join, isdir
import os
from os import listdir
from scipy import misc, ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import math

__ending__ = "_cropped.bmp"  # ending of filename for files to be processed


def to_8bit(im):
    im = np.mean(im, axis=2)
    return im


parser = argparse.ArgumentParser()
parser.add_argument('path', help="Path to directory with images")
parser.add_argument('cal', help="mm/pixel in images", type=float)
parser.add_argument('-d', help="overlay normal distributions", action="store_true")
args = parser.parse_args()
cal = args.cal
path = args.path
plot_distr = args.d

files = [f for f in listdir(path) if (isfile(join(path, f)) and f.__contains__(__ending__))]

data_path = join(path, "data/")

# only create a new directory if there are files to process
if len(files) > 0:
    if not isdir(data_path):
        os.makedirs(data_path)

output = open(join(data_path, "radii.csv"), mode='w')
output.write("File name,2rms X radius (mm),2rms Y radius (mm)\n")

for f in files:
    print "Processing " + f
    img = misc.imread(join(path, f))
    full_img = misc.imread(join(path, f[:-len(__ending__)]) + ".bmp")  # uncropped photo

    full_img_blurred = ndimage.filters.gaussian_filter(full_img, 2)
    img_blurred = ndimage.filters.gaussian_filter(img, 2)
    f_brightest = np.unravel_index(np.argmax(full_img_blurred), full_img.shape)
    brightest = np.unravel_index(np.argmax(img_blurred), img.shape)
    offset = np.array(
        [f_brightest[0] - brightest[0], f_brightest[1] - brightest[1]])  # offset between cropped and full images

    max_val = np.amax(img)
    min_val = np.amin(img)
    img -= min_val  # filter out background noise

    img_sum = np.sum(img)
    i_avg = 0  # x centroid
    j_avg = 0  # y centroid
    for (j, i), v in np.ndenumerate(img):
        i_avg += (i * v)  # x moment
        j_avg += (j * v)  # y moment
    i_avg /= img_sum
    j_avg /= img_sum

    centroid = np.array([i_avg, j_avg])
    center = np.array([306.0, 247.0])
    abs_centroid = centroid + offset
    skew = (abs_centroid - center) * cal
    print abs_centroid

    i_ssr = 0   #sum squared residuals
    j_ssr = 0
    for (j, i), v in np.ndenumerate(img):
        i_ssr += (i - i_avg) ** 2 * v
        j_ssr += (j - j_avg) ** 2 * v
    x_rms = np.sqrt(i_ssr / img_sum) * cal  #also the standard deviation
    y_rms = np.sqrt(j_ssr / img_sum) * cal

    #calculate normal distributions
    mu_x = 0
    x_distr = np.linspace(-3*x_rms, 3*x_rms, 100)   #3 standard deviations
    x_distr_plot = mlab.normpdf(x_distr, mu_x, x_rms)   #Normal distribution
    x_multiplier = 1/x_distr_plot[len(x_distr_plot)/2]  #scale so max=1
    mu_y = 0
    y_distr = np.linspace(-3*y_rms, 3*y_rms, 100)
    y_distr_plot = mlab.normpdf(y_distr, mu_y, y_rms)
    y_multiplier = 1/y_distr_plot[len(y_distr_plot)/2]

    # plot intensity distributions along axes that go through the beam centroid (should be K-V-like)
    p = round(i_avg)
    q = round(j_avg)
    y_cent_row = img[:, q]
    x_cent_col = img[p, :]
    plt.figure(1)
    plt.subplot(121)
    plt.plot((range(y_cent_row.shape[0]) - j_avg) * cal,
             y_cent_row / float(max_val-min_val))  # distance from x centroid vs. relative intensity
    if plot_distr:
        plt.plot(x_distr, x_distr_plot*x_multiplier)
    plt.title("Y")
    plt.xlabel("Distance from y centroid (mm)")
    plt.ylabel("Relative intensity")
    plt.subplot(122)
    plt.plot((range(x_cent_col.shape[0]) - i_avg) * cal, x_cent_col / float(max_val-min_val))   #normalize so the max value=1
    if plot_distr:
        plt.plot(y_distr, y_distr_plot*y_multiplier)
    plt.title("X")
    plt.xlabel("Distance from x centroid (mm)")
    plt.ylabel("Relative intensity")
    plt.savefig(join(data_path, f[:-4] + "_plots.png"))  # the -4 removes the ".bmp" extension
    plt.close()

    '''
    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    x_coords = []
    y_coods = []
    z = []
    for (j,i),v in np.ndenumerate(img):
        x_coords.append(i)
        y_coods.append(j)
        z.append(v)

    ax.scatter(x_coords,y_coods,z,c='r',marker='o')
    plt.show()
    '''
    output.write("{0},{1},{2}\n".format(f, 2 * x_rms, 2 * y_rms))

if len(files) > 0:
    output.close()
