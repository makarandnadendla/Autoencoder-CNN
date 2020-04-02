# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
	@staticmethod
	def build(width, height, depth):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# define the input to the encoder
		encoderInputs = Input(shape=inputShape)
		x = encoderInputs

		# build the layers
		x = (Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)      # 16x16x32
		x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)      # 16x16x32
		x = BatchNormalization(axis=chanDim)(x)     # 16x16x32
		
		# build the encoder model
		encoder = Model(encoderInputs, x, name="encoder")

		# define the input to the decoder
		decoderInputs = Input(shape = encoder.layers[-1].output_shape[1:])
		x = decoderInputs

		# build the layers
		x = UpSampling2D()(x)
		x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)      # 32x32x32
		x = BatchNormalization(axis = chanDim)(x)
		x = (Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))(x)   # 32x32x3
		
		# build the decoder model
		decoder = Model(decoderInputs, x, name = "decoder")

		# our autoencoder is the encoder + decoder
		autoencoder = Model(encoderInputs, decoder(encoder(encoderInputs)),
			name="autoencoder")

		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)