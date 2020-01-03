# STRUCTURE: 
# - Uses exclusively 3x3 CONV filters, similar to VGGNet
# - Stacks multiple 3x3 CONV filters on top of each other prior to performing max-pooling (again, similar to VGGNet)
# - Unlike VGGNet, uses depthwise separable convolution rather than standard convolution layers (https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
	@staticmethod
	# Parameter list:
	# - width, height: image size
	# - depth: the number of color channels each image contains
	# - classes: the number of classes our network will predict (2 by default)
	# Return: the constructed network architecture
	def build(width, height, depth, classes=2):
		# initialize the model
		model = Sequential()
		
		# "channels first" and "channels last"
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1	
		elif K.image_data_format() == "channels_last":
			inputShape = (height, width, depth)
			chanDim = -1

		# CONV => RELU => POOL
		model.add(SeparableConv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU => POOL) * 2
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU => POOL) * 3
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# FC => RELU layers: head of the network
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model