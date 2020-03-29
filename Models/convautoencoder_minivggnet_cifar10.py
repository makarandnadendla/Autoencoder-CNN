# USAGE
# python convautoencoder_minivggnet_cifar10.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from modelcollection.nn.conv import AutoencoderMiniVGGNet
from modelcollection.preprocessing import CifarDataPreprocessing
from modelcollection.plot import PlotCm, PlotCr
from modelcollection.callbacks import TrainingMonitor
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type = str, default = 'output/plots/convautoencoder_minivggnet',
	help="path to the output plot folder")
ap.add_argument("-w", "--weights",type = str, default = 'output/weights/convautoencoder_minivggnet/convautoencoder_minivggnet_cifar10_best_weights.hdf5',
	help="path to best model weights file")
ap.add_argument("-a", "--autoencoder",type = str, default = 'output/weights/conveautoencoder/convautoencoder_cifar10_best_weights.hdf5',
	help="path to best autoencoder model weights file")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}".format(os.getpid()))

#Set epochs and batch num
epochs = 200
batchSize = 64

# load the training and testing data, reduce bird, deer and truck to 50%
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX, trainY = CifarDataPreprocessing(trainX, trainY, [2,4,9], randomSeed = 101)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")

# scale the data into the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
autoenc_model = load_model(args["autoencoder"])
opt = SGD(lr=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
model = AutoencoderMiniVGGNet.build(width=32, height=32, depth=3, classes=10, autoencoder = autoenc_model)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss
figPath = os.path.sep.join([args["output"], "cifar10_convautoencoder_minivggnet.png"])
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss",
	save_best_only=True, verbose=1)
callbacks = [TrainingMonitor(figPath), checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=batchSize),
		validation_data=(testX, testY), epochs=epochs, callbacks = callbacks, 
		steps_per_epoch=len(trainX) // batchSize, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batchSize)
PlotCr(testY, predictions, target_names=labelNames, output_path = os.path.sep.join([args["output"], "cifar10_convautoencoder_minivggnet_classification_report.png"]))
PlotCm(testY, predictions, target_names=labelNames, output_path = os.path.sep.join([args["output"], "cifar10_convautoencoder_minivggnet_conf_matrix.png"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(os.path.sep.join([args["output"], "cifar10_convautoencoder_minivggnet.png"]))