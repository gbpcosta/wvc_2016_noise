import os
import cv2
import numpy as np

import sys, argparse

parser = argparse.ArgumentParser(description='Script for insert noise in images (Gaussian and Poisson).')
parser.add_argument('-in','--input', help='Input PATH with the input images (Color images)',required=True)
parser.add_argument('-type','--noiseType',help='Choose which noise will apply (gaussian or poisson)', choices=['gaussian', 'poisson'], required=True)
parser.add_argument('-sig','--sigma',help='Input the parameter Sigma (Noise value for Gaussian noise)', type=int, required=False, default=10)
parser.add_argument('-lam','--lambdav',help='Input the parameter Lambda (Noise value for Poisson noise)', type=int, required=False, default=1)

args = parser.parse_args()
pathOriginalImages = args.input
typeNoise = args.noiseType

# Path of dataset
os.chdir(pathOriginalImages)
path, filename = os.path.split(os.getcwd())

# Check type noise and create folder
if typeNoise == 'gaussian':
    pathNew = os.path.join(path, 'gaussian-'+str(args.sigma))
elif typeNoise == 'poisson':
    pathNew = os.path.join(path, 'posson-'+str(args.lambdav))

if not os.path.exists(pathNew):
    os.makedirs(pathNew)

for di in os.listdir(pathOriginalImages):
    # Original path with subdirectories
    pathOri = os.path.join(pathOriginalImages, di)

    #Get new path noise images
    path = os.path.join(pathNew, di)

    #Check subdirectory and create if dont exist
    if not os.path.exists(path):
        os.makedirs(path)

    frameList = sorted(os.listdir(pathOri))
    frameList = [f for f in frameList if '.png' or '.jpg' in f]
    for f in frameList:
        originalImagePath = os.path.join(pathOri, f)
        original = cv2.imread(originalImagePath)
        original = original.astype(np.float64)

        if typeNoise == 'gaussian':
            noise = np.random.normal(0, args.sigma, original.shape).astype(float)
        elif typeNoise == 'poisson':
            noise = np.random.poisson(args.lambdav,original.shape).astype(float)
            noise = (noise - np.mean(noise))

        noiseImage = original + noise

        noiseImage[noiseImage > 255.0] = 255.0
        noiseImage[noiseImage < 0.0] = 0.0
        noiseImage = noiseImage.astype(np.uint8)

        noiseImagePath = path + '/' + f
        print noiseImagePath
        cv2.imwrite(noiseImagePath, noiseImage)
