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
    
 - Requirements
 - Directory Structure
 - Tests
 - Code Example
 - References
 - What's Next?
 
 
 ```python
python convautoencoder_cifar10.py
```
<details>
<summary>Optional Arguments</summary>
"--samples" "# number of samples to visualize when decoding"<br>
"--image" "path to output image comparison file"<br>
"--output" "path to output plot file"<br>
"--weights" "path to best model weights file"
</details>
