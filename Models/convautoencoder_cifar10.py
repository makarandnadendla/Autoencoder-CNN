# USAGE
# python convautoencoder_cifar10.py 

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from modelcollection.nn.conv import ConvAutoencoder
from modelcollection.preprocessing import CifarDataPreprocessing
from modelcollection.callbacks import TrainingMonitor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-i", "--image", type=str, default="output/plots/convautoencoder/autoencoder_only_output.png",
	help="path to output image comparison file")
ap.add_argument("-o", "--output", type=str, default="output/plots/convautoencoder/autoencoder_only_plot.png",
	help="path to output plot file")
ap.add_argument("-w", "--weights",type = str, default = 'output/weights/convautoencoder/convautoencoder_cifar10_best_weights.hdf5',
	help="path to best model weights file")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}".format(os.getpid()))

# initialize the number of epochs to train for and batch size
epochs = 5
batchSize = 32

# load the training and testing data, reduce bird, deer and truck to 50%
print("[INFO] loading CIFAR_10 dataset...")
((trainX, trainY), (testX, _)) = cifar10.load_data()
trainX, trainY = CifarDataPreprocessing(trainX, trainY, [2,4,9], randomSeed = 101)

# scale the data into the range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(width=32, height=32, depth=3)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)

# construct the callback to save only the *best* model to disk
# based on the validation loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	callbacks = callbacks, 
	epochs=epochs,
	batch_size=batchSize)

# construct a plot that plots and saves the training history
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["output"])

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
outputs = None

# loop over our number of output samples
for i in range(0, args["samples"]):
	# grab the original image and reconstructed image
	original = (testX[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")

	# stack the original and reconstructed image side-by-side
	output = np.hstack([original, recon])

	# if the outputs array is empty, initialize it as the current
	# side-by-side image display
	if outputs is None:
		outputs = output

	# otherwise, vertically stack the outputs
	else:
		outputs = np.vstack([outputs, output])

# save the outputs image to disk
cv2.imshow(args["image"], outputs)
cv2.waitKey(0)
cv2.imwrite(args["image"], outputs)