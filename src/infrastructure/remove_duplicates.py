import logging
import os

import matplotlib.pyplot as plt
from PIL import Image

import src.config.scrap_config_template as config

os.chdir(config.PATH_IMAGE_DUPLICATES)


files_list = os.listdir(".")

logging.info(f"{len(files_list)} files")

duplicates = []
hash_keys = dict()

for index, filename in enumerate(os.listdir(".")):
    if not filename.startswith("."):
        if os.path.isfile(filename):
            img = Image.open(filename)
            # reduce image:
            img = img.resize((6, 6), Image.ANTIALIAS)
            # reduce color
            img = img.convert("L")
            # find average pixel value
            pixel_data = list(img.getdata())
            avg_pixel = sum(pixel_data) / len(pixel_data)
            # Compute hash in base 16:
            bits = "".join(["1" if (px >= avg_pixel) else "0" for px in pixel_data])
            filehash = str(hex(int(bits, 2)))[2:][::-1].upper()

            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index, hash_keys[filehash]))

logging.info(f"{len(duplicates)} duplicates have been found")

for file_indexes in duplicates[:30]:
    try:
        plt.subplot(121), plt.imshow(plt.imread(files_list[file_indexes[1]]))
        plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(plt.imread(files_list[file_indexes[0]]))
        plt.title(str(file_indexes[0]) + "duplicate"), plt.xticks([]), plt.yticks([])
        plt.show()

ans = input("Do you want to delete the duplicates? [y/n]")

if ans == "y":
    for index in duplicates:
        os.remove(files_list[index[0]])
else:
    print("Nothing deleted, please review pictures manually or run the script again")
