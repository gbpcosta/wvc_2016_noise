import cv2
import os
import numpy as np

dataset = "/home/gbpcosta/datasets/CorelDB"
classes = next(os.walk(dataset))[1]

for i in classes
    img_dir = '%s/%s' % (dataset, i)
    img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir)]

    for
