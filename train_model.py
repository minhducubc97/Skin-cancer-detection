# import the necessary packages
import matplotlib
from library.cancernet import CancerNet
from library import config
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
matplotlib.use("Agg")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="graph.png")
args = vars(ap.parse_args())

# initialize the number of epochs, initial learning rate, and batch size
NUM_EPOCHS = 40
INIT_LEARNRATE = 0.1
BATCH_SIZE = 32

# calculate the total number of image paths in training, validation, and testing directories
trainPaths = list(paths.list_images(config.TRAIN_PATH))
numTrain = len(list(paths.list_images(config.TRAIN_PATH)))
numTest = len(list(paths.list_images(config.TEST_PATH)))
numVal = len(list(paths.list_images(config.VAL_PATH)))

# account for skew in the labeled data
trainLabels = [int(path.split(os.path.sep)[-2]) for path in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max()/classTotals

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# initialize the validation and testing data augmentation object
valAug = ImageDataGenerator(rescale=1/255)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE
)

# initialize the testing generator 
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3, classes=2)
opt = Adagrad(learning_rate=INIT_LEARNRATE, decay=INIT_LEARNRATE/NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training process to fit the model
H = model.fit_generator(
    trainGen,
    steps_per_epoch=numTrain//BATCH_SIZE, # integer division
    validation_data=valGen,
    validation_steps=numVal//BATCH_SIZE,
    class_weight=classWeight,
    epochs=NUM_EPOCHS
)

# reset the testing generator and use the trained model to make predictions on the data
print("[INFO] Evaluating network ...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(numTest//BATCH_SIZE)+1)

# find the index of the label with the highest corresponding probability of each image in the test set
predIdxs = np.argmax(predIdxs, axis=1)

# display the classification report
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

# create the confusion matrix and calculate the raw accuracy, sensitivity, specificity
confMatrix = confusion_matrix(testGen.classes, predIdxs)
totalValue = sum(sum(confMatrix))
accuracy = (confMatrix[0,0]+confMatrix[1,1])/totalValue
sensitivity = confMatrix[0,0]/(confMatrix[0,0]+confMatrix[0,1])
specificity = confMatrix[1,1]/(confMatrix[1,0]+confMatrix[1,1])

# display the confusion matrix, accuracy, sensitivity, and specificity
print(confMatrix)
print("accuracy: {:.3f}".format(accuracy))
print("sensitivity: {:.3f}".format(sensitivity))
print("specificity: {:.3f}".format(specificity))

# plot the training loss and accuracy
print("[INFO] Generating result graph ...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.legend(loc="lower left")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.savefig(args["plot"])

# finish the model
print("[INFO] Deep Learning model is complete!")