Facade Generation using Pix2Pix GAN


Project Overview
This project implements a Pix2Pix Generative Adversarial Network (GAN) for generating architectural facades. 

The model is capable of transforming input sketches or simplified drawings into photorealistic facade images, demonstrating the power of image-to-image translation using deep learning techniques.
Features
Image-to-image translation using Pix2Pix GAN architecture
Generates realistic architectural facade images from input sketches
Supports custom dataset training and inference
Includes data preprocessing and augmentation scripts
Visualizes generation results and training progress

Prerequisites

Python 3.8+
CUDA-enabled GPU 
Following libraries:

TensorFlow
Keras
NumPy
Matplotlib


Data Structure
Copy/data
├── train
│   ├── input
│   └── target
├── test
│   ├── input
│   └── target
└── validation
    ├── input
    └── target
