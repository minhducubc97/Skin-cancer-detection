# import the necessary packages
from library import config
from imutils import paths
import random
import shutil
import os

# grab the paths to all input images in the original input directory
# and shuffle them
imagePaths = list(paths.list_images(config.ORIGINAL_DATASET))
random.seed(50)
random.shuffle(imagePaths)

# compute the training and testing split
numImages = len(imagePaths)
numTrain = int(numImages*config.TRAIN_RATIO)
numTest = int(numImages*config.TEST_RATIO)
trainPaths = imagePaths[:numTrain]
testPaths = imagePaths[numTrain:numTrain+numTest]
valPaths = imagePaths[numTrain+numTest:]

# define the datasets that we'll be building
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image and extract the class label
		filename = inputPath.split(os.path.sep)[-1]
		labelString = inputPath.split(os.path.sep)[-2]

		# convert label string to binary code: "0" for "benign" and "1" for "malignant"
		label = 0
		if (labelString == "benign"):
			label = 0
		elif (labelString == "malignant"):
			label = 1

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, str(label)])

		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct the path to the destination image and then copy the image itself
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)