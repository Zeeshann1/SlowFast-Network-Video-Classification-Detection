# SlowFast Network Video Classification & Detection
#### Slowfast AI algorithm recognized what activity is being performed in the video. It also can detect any action is happening. Used for video understanding research on different tasks (classification, detection, &amp; etc.)

## Features

-	This algorithm is from Facebook AI research 
-	PySlowFast: high performance, light weight and efficient video understanding codebase in pytorch.
-	SlowFast networks pre-trained on the Kinetics 400 dataset
-	Slowfast AI algorithm recognized what activity is being performed in the video. It also can detect any action is happening 
-	Used for video understanding research on different tasks (classification, detection, & etc.).
- SlowFast: novel method to analyze the contents of a video segment.
-	It has two path away, one is capturing semantics of the object which is class slow path away another one is fast path away which capture the motion.
-	Both paths performs 3D convolution operation 
-	Github: https://github.com/facebookresearch/SlowFast

## Backbone model architectures:
-	SlowFast
-	Slow
-	C2D
-	I3D
-	Non-local Network
-	X3D
-	MViTv1 and MViTv2

## Kinetics Dataset

-	SlowFast networks pre-trained on the Kinetics 400 dataset
-	Kinetics: datasets of URL links of up to 650,000 video clips that cover 400/600/700 human action classes, depending on the dataset version
-	Videos include: human-object interactions, human-human interactions such as shaking hands and hugging
-	Each action class has at least 400/600/700 video clips
-	Each clip is human annotated with a single action class and lasts around 10 seconds.
-	Load a pre trained video classification model in PyTorchVideo and run it on a test video
-	Running SlowFast networks pre-trained on the Kinetics 400 dataset.
-	Link: https://www.deepmind.com/open-source/kinetics

## Installation

Create new conda environment run on jupyter notebook or google colab notebook
```bash
conda create --name Slowfast
git clone https://github.com/facebookresearch/SlowFast.git

```
- Install all required libraries and packages from
   https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md

## Perform the Code
-	Import all functions
-	Load the model
-	Set the model to eval mode and move to desired device
-	Download the id to label mapping for the Kinetics 400 dataset on which the torch hub models were trained.
-	This will be used to get the category label names from the predicted class ids.
-	Define input transform
-	Before passing the video into the model we need to apply some input transforms and sample a clip of the correct duration
-	Run Inference
-	Download an example video
-	Load the video and transform it to the input format required by the model
-	Get Predictions

## SlowFast Model by FAIR PyTorchVideo
-	SlowFast model architectures are based on [1] with pre-trained weights using the 8x8 setting on the Kinetics dataset.
-	Both the Slow and Fast pathways use a 3D ResNet model, capturing several frames at once and running 3D convolution operations on them
-	Reference: [1] Christoph Feichtenhofer et al, “SlowFast Networks for Video Recognition” https://arxiv.org/pdf/1812.03982.pdf
#### Model Implementation: Google Colab Notebook Code Link Given Below
-	Link:[ https://colab.research.google.com/drive/1xw0SCaCRTN-KaPAZeLY_lPRnmctdPSNx](https://colab.research.google.com/drive/1NKx1RH8sZzJQ88T3_dTDbWKIZjSFOlYF)

![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/1.png)

## 3D RESNET on Slow Architecture Network
-	The model architecture is based on [1] with pre-trained weights using the 8x8 setting on the Kinetics dataset.
![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/2.png)
-	A residual neural network (ResNet) is an artificial neural network (ANN)
-	ResNet 3D is a type of model for video that employs 3D convolutions.
-	This model collection consists of two main variants.
-	 The first formulation is named mixed convolution (MC) and consists in employing 3D convolutions only in the early layers of the network, with 2D convolutions in the top layers.
#### Model Implementation: Google Colab Notebook Code Link Given Below
-	Link: https://colab.research.google.com/drive/1NKx1RH8sZzJQ88T3_dTDbWKIZjSFOlYF

## X3D Model by FAIR PyTorchVideo
-	X3D model architectures are based on pre-trained on the Kinetics dataset
![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/3.png)
- Reference Paper: Christoph Feichtenhofer, “X3D: Expanding Architectures for Efficient Video Recognition.” https://arxiv.org/abs/2004.04730
  
#### X3D: Expanding Architectures for Efficient Video Recognition

-	X3D is Multi-grid Training:
-	Multi-grid training is a mechanism to train video architectures efficiently. Instead of using a fixed batch size for training, this method proposes to use varying batch sizes in a defined schedule, yet keeping the computational budget approximately unchanged by keeping batch x time x height x width a constant.
![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/4.png)

#### Model Implementation: Google Colab Notebook Code Link Given Below
- Link: https://colab.research.google.com/drive/1rd5c0H9dzL5C7ltVQ2IPEa1aRZUEvF8D#scrollTo=rVa8kxbN9-aP

## Demo
[![Watch the video](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/archery.mp4)

## Predicted Results
![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/result.png)


## Results comparison

![Logo](https://github.com/Zeeshann1/SlowFast-Network-Video-Classification-Detection/blob/main/5.png)




## References
- https://github.com/facebookresearch/SlowFast
- https://towardsdatascience.com/slowfast-explained-dual-mode-cnn-for-video-understanding-8bf639960256
- https://arxiv.org/abs/2004.04730
- https://arxiv.org/pdf/1812.03982.pdf
- https://colab.research.google.com/drive/1rd5c0H9dzL5C7ltVQ2IPEa1aRZUEvF8D#scrollTo=rVa8kxbN9-aP
- https://colab.research.google.com/drive/1NKx1RH8sZzJQ88T3_dTDbWKIZjSFOlYF
- https://colab.research.google.com/drive/1NKx1RH8sZzJQ88T3_dTDbWKIZjSFOlYF
-  https://www.deepmind.com/open-source/kinetics
