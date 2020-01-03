# import the necessary packages
import os

ORIGINAL_DATASET = "datasets/original"

# training, validation and testing directories
TRAIN_PATH = "datasets/working-dataset/train"
VAL_PATH = "datasets/working-dataset/validate"
TEST_PATH = "datasets/working-dataset/test"

# amount of data for training
TRAIN_SPLIT = 0.8

# amount of data for validation (as a percentage of the training data)
VAL_SPLIT = 0.1