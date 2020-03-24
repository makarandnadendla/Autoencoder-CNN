# Autoencoder-CNN

The aim of this project is to evaluate a simple hypothesis:
    
"Can an Autoencoder + Convolutional Neural Network approach perform better than a simple Convolutional Neural Network classifier?"

Autoencoders are neural networks that traditionally look to encode raw pixel data into a few nodes of a hidden layer, and then decode, or reconstruct the image from that 'compression'     

The idea behind the hypothesis is that, the autoencoder might learn to emphasize key distinguishing features when it is trained. Therefore, if we use the encoded representation of the image as input to our CNN, we might achieve better results than if we used the raw image pixel data as input. 
    
## Motivation

The motivation behind this project, was simply to explore what autoencoders can and cannot provide to other algorithms and evaluate their usefulness in deepening any particular algorithm. 
    
Autoencoders are a popular algorithm for beginner deep learning practitioners because the idea is easy to understand, and it has a aesthetic appeal to it where doing something so simple might yield benefits to a classification algorithm in question. 
    
However, as with any other hypothesis, we must evaluate it without bias, using the most relevant performance metrics (precision, accuracy, f-score, confusion matrix) in proper context to determine the efficacy. 

Mechanistic speculation of the algorithm, by itself, cannot lead to more objective conclusions, which is why we are testing the algorithm today.
    
 
## Table Of Contents

 - Overview
 - Requirements
 - Directory Structure
 - Tests
 - Code Example
 - References
 - What's Next?
 
 ## Overview

We'll be exploring 5 different models and their performance on CIFAR-10 in this project:
    
 - Convolutional Autoencoder 
 - MiniVGGNet
 - ShallowNet
 - Encoder + MiniVGGNet
 - Encoder + ShallowNet
    
In the case of the Convolutional Autoencoder, we'll be testing it's ability to recreate images from the CIFAR-10 dataset, and based on those results we'll use the encoder layers as inputs for both MiniVGGNet and ShallowNet. For the rest, we'll be evaluating the model on their ability to correctly classify images in our test set.
    
## Requirements

 The relevant libraries are listed below: and their versions are listed below. I used a conda virtual environment to run this project, so all the installation instructions will be presumed to be in a conda environment.
- conda 4.8.3
- conda-build 3.18.8
- python 3.7.6
- tensorflow-gpu 2.1 (tensorflow 2.1 should also work here just fine)
- matplotlib 3.1.3
- numpy 1.18.1    
- pandas 1.02
- scikit-learn 0.21.3
- argparse 1.3.0 
- opencv 4.2.0
- seaborn 0.10.0

You can install Anaconda from https://www.anaconda.com/distribution/, and then run the following commands:
    
```python
conda install -c conda-forge python=3.7.6
conda install -c anaconda tensorflow=2.1
conda install -c conda-forge matplotlib=3.1.3 
conda install -c conda-forge numpy=1.18.1
conda install -c anaconda pandas=1.02
conda install scikit-learn=0.21.3
conda install -c anaconda argparse=1.3.0
conda install -c conda-forge opencv=4.2.0
conda install -c anaconda seaborn=0.10.0
```
 
 ## Directory Structure

```bash
C:.
└───Models
    │   convautoencoder_cifar10.py
    │   convautoencoder_minivggnet_cifar10.py
    │   convautoencoder_shallownet_cifar10.py
    │   minivggnet_cifar10.py
    │   shallownet_cifar10.py
    │
    ├───modelcollection
    │   ├───callbacks
    │   ├───nn
    │   │   └───conv
    │   │        │   convautoencoder.py
    │   │        │   convautoencoder_minivggnet.py
    │   │        │   convautoencoder_shallownet.py
    │   │        │   minivggnet.py
    │   │        └───shallownet.py 
    │   ├───plot
    │   └───preprocessing
    │
    └───output
        ├───plots
        │   ├───convautoencoder_minivggnet
        │   ├───convautoencoder_shallownet
        │   ├───conveautoencoder
        │   ├───minivggnet
        │   └───shallownet
        └───weights
            ├───convautoencoder_minivggnet
            ├───convautoencoder_shallownet
            ├───conveautoencoder
            ├───minivggnet
            └───shallownet
```

The main scripts are in the Models folder, with each of the models having it's own script. 

Each of the respective models' classes are in Models/modelcollection/nn/conv if you need to take a look at or alter the structure of the networks.
 
 
## Tests 

### Convolutional Autoencoder

```python
python convautoencoder_cifar10.py
```
<details>
<summary>Optional Arguments</summary>


- --samples
- number of samples to visualize when decoding, 
- default:8
<br>
    
- --image
- path to output image comparison file 
- default="output/plots/conveautoencoder/autoencoder_only_output.png"
<br>

- --output
- path to output plot file
- default="output/plots/conveautoencoder/autoencoder_only_plot.png"
<br>

- --weights
- path to best model weights file
- default = 'output/weights/conveautoencoder/convautoencoder_cifar10_best_weights.hdf5'
<br>
</details>

### Convolutional Autoencoder + MiniVGGNet

```python
python convautoencoder_minivggnet_cifar10.py
```
<details>
<summary>Optional Arguments</summary>


- --output
- path to the output loss/accuracy plot
- default="output/plots/convautoencoder_minivggnet"
<br>

- --weights
- path to best model weights file
- default = 'output/weights/convautoencoder_minivggnet/convautoencoder_minivggnet_cifar10_best_weights.hdf5'
<br>
    
- --autoencoder
- path to best autoencoder model weights file
- default = 'output/weights/conveautoencoder/convautoencoder_cifar10_best_weights.hdf5'
<br>
</details>

### Convolutional Autoencoder + ShallowNet

```python
python convautoencoder_shallownet_cifar10.py
```
<details>
<summary>Optional Arguments</summary>


- --output
- path to the output loss/accuracy plot
- default="output/plots/convautoencoder_shallownet"
<br>

- --weights
- path to best model weights file
- default = 'output/weights/convautoencoder_shallownet/convautoencoder_shallownet_cifar10_best_weights.hdf5'
<br>
    
- --autoencoder
- path to best autoencoder model weights file
- default = 'output/weights/conveautoencoder/convautoencoder_cifar10_best_weights.hdf5'
<br>
</details>

### MiniVGGNet

```python
python shallownet_cifar10.py
```
<details>
<summary>Optional Arguments</summary>

  
- --output
- path to the output loss/accuracy plot
- default= "output/plots/minivggnet"
<br>

- --weights
- path to best model weights file
- default = 'output/weights/minivggnet/minivggnet_cifar10_best_weights.hdf5'
<br>
</details>

### ShallowNet

```python
python shallownet_cifar10.py
```
<details>
<summary>Optional Arguments</summary>

    
- --output
- path to the output loss/accuracy plot
- default= "output/plots/shallownet"
<br>

- --weights
- path to best model weights file
- default = 'output/weights/shallownet/shallownet_cifar10_best_weights.hdf5'
<br>
</details>
