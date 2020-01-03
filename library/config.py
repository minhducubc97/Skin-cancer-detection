# import the necessary packages
import os

ORIGINAL_DATASET = "datasets/skin-cancer-malignant-vs-benign"

# training, validation and testing directories
TRAIN_PATH = "datasets/working-dataset/train"
VAL_PATH = "datasets/working-dataset/validate"
TEST_PATH = "datasets/working-dataset/test"

# amount of data for training (75%), validation (5%) and testing (20%)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.05
TEST_RATIO = 0.2