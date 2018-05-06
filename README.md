# CNN_DOG
Author Tianyi Zhao

This work is inspired by Kaggle Playground Prediction Competition: Dog Breed Identification 
(https://www.kaggle.com/c/dog-breed-identification)

# Database
Contents of Stanford Dogs dataset:

Number of categories: 120

Number of images: 20,580

Annotations: Class labels, Bounding boxes


Training set: Contains 12,000 images, 120 categories, 100 images for each category.

Test Set: Contains 8,580 images, 120 categories.

# Implementation
Use VGG19 which is pre-trained on ImageNet and transfer this model for our dog breed classification problem.

vgg_trainable.py: Construction of VGG19 network

Demo.py: Training VGG19

Test_Visualization.ipynb: Get visualization of our model.

Test.ipynb: Get result visualization on test set.
# Experiment Setting:
batch size = 64

Initial learning rate 0.001, decay step = 100, decay rate = 0.9

Activation function: ReLu

Dropout: 0.5
