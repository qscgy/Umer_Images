import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import isfile, join, isdir
import os
from os import listdir
from scipy import misc, ndimage
from mpl_toolkits.mplot3d import Axes3D

__ending__ = "_cropped.bmp"  # ending of filename for files to be processed


def to_8bit(im):
    im = np.mean(im, axis=2)
    return im


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Name of the image to plot")
parser.add_argument('-a', help="create animated frames", action="store_true")
args = parser.parse_args()
filename = args.filename
animate = args.a


print "Processing " + filename
img = misc.imread(filename)

max_val = np.amax(img)
min_val = np.amin(img)
img -= min_val  # filter out background

figure = plt.figure()
ax = Axes3D(figure)
x_coords = []
y_coords = []
z = []
for (j,i),v in np.ndenumerate(img):
    if j%3==0 and i%3==0:   #reduce the number of points to plot
        x_coords.append(i)
        y_coords.append(j)
        z.append(v)

#ax.scatter(x_coords,y_coords,z,c='r',marker='o')
ax.plot_wireframe(x_coords, y_coords, z, rstride=40, cstride=40)
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
ax.set_zlabel('Intensity')
save_path = "/Users/sam/Documents/animations/"
#figure.savefig("/Users/sam/Documents/animations/movie_test.png")
if animate:
    for ii in xrange(0,360,20):
        ax.view_init(elev=10., azim=ii)
        file_str = "movie{0}.png".format(ii)
        file_path = os.path.join(save_path, file_str)
        plt.savefig(file_path)
plt.show()
