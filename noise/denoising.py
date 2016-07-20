import os
import cv2
import numpy as np
import sys, argparse
import skimage
from skimage.morphology import disk
from skimage.filters.rank import median

parser = argparse.ArgumentParser(description='Script for apply denoising methods in images.')
parser.add_argument('-in','--input', help='Input PATH with the input images',required=True)
parser.add_argument('-m', '--method', required=True, default="NLM", help="", choices=['NLM','Bilateral','Median'])
parser.add_argument('-l', '--level', required=True, help="",type=int)
parser.add_argument('-t', '--transform', help="Anscombe tranform",type=bool,choices=[False,True],default=False)

args = parser.parse_args()
pathImages = args.input

# Path of dataset
os.chdir(pathImages)
path, filename = os.path.split(os.getcwd())

# Create folder
pathNew = os.path.join(path, filename+"-"+args.method+"-"+str(args.level))
if not os.path.exists(pathNew):
    os.makedirs(pathNew)

for di in os.listdir(pathImages):
    # Original path with subdirectories
    pathOri = os.path.join(pathImages, di)

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

        #Anscombe transformation
        if args.transform:
            print "aqui"
            original = 2*np.sqrt(original.astype(np.float) + 3/8)
            original = original.astype(np.uint8)

        if args.method == 'NLM':
            denoisedImage = cv2.fastNlMeansDenoisingColored(original,None,args.level,args.level,7,21)
            #denoisedImage = denoise_nl_means(a, 7, 11, 0.1,False)
        elif args.method == 'Bilateral':
            denoisedImage = cv2.bilateralFilter(original,args.level,100,100)
        elif args.method == 'Median':
            denoisedImage = cv2.medianBlur(original, args.level)

        #Anscombe inverse transformation
        if args.transform:
            denoisedImage = np.power((denoisedImage/2),2) - 3/8

        denoisedImage[denoisedImage > 255.0] = 255.0
        denoisedImage[denoisedImage < 0.0] = 0.0


        denoisedImagePath = path + '/' + f
        print denoisedImagePath
        cv2.imwrite(denoisedImagePath, denoisedImage)
