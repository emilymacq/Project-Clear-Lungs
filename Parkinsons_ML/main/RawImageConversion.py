
# Import packages
import glob
import math
import numpy as np
import rawpy
import os.path
from os.path import dirname, abspath

# Import methods from other scripts


# Constant Declarations
path_to_raw_image_folder = os.path.join(dirname(dirname(abspath(__file__))), os.path.join('resources', os.path.join('input', 'raw_images')))


def convert_raw_images_to_csv():
    raw_image_converter = RawImageConvert()
    raw_image_paths = raw_image_converter.get_paths_to_raw_images()

    for raw_image_path in raw_image_paths:
        image_data = raw_image_converter.read_raw_image(raw_image_path)
    return

class RawImageConvert:

    def __init__(self):
        return

    def get_paths_to_raw_images(self):
        image_path_list = glob.glob(os.path.join(path_to_raw_image_folder, '*'))
        return image_path_list


    def read_raw_image(self, raw_image_path):
        image_data = np.fromfile(raw_image_path, dtype='int16', sep="")

        idx = image_data[image_data > (pow(2, 15))]
        for index in idx:
            image_data[index] = image_data[index] - (pow(2, 16))

        n = 256
        m1 = np.empty([1, 1])
        m2 = np.empty([1, 1])
        m3 = np.empty([1, 1])
        m4 = np.empty([1, 1])
        L = 4 * n * math.floor(len(image_data) / (4 * n))

        print(L)
        for i in range(0, 500):
            print(i)
            m1 = np.append(m1, image_data[(i + 0 * n): (i + 1 * n - 1)])
            m2 = np.append(m2, image_data[(i + 1 * n): (i + 2 * n - 1)])
            m3 = np.append(m3, image_data[(i + 2 * n): (i + 3 * n - 1)])
            m4 = np.append(m4, image_data[(i + 3 * n): (i + 4 * n - 1)])
            i += n * 4

        t = np.arange(len(m1)) / 44100

        return image_data


if __name__ == '__main__':
    convert_raw_images_to_csv()

