# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

class AutoencoderShallowNet:
	@staticmethod
	def build(width, height, depth, classes, autoencoder):
		
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()

		model.add(Input(shape=inputShape))

		encoder = autoencoder.layers[1]
		encoder.trainable = False
		model.add(encoder)

		# define the first (and only) CONV => RELU layer
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model