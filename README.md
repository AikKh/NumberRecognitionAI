# Digit Recognition Neural Network

A simple digit recognition neural network implemented in C++ using STL. The network features customizable layers, Leaky ReLU activation, and efficient training using the MNIST dataset with batch processing. Basic visualization is supported using SFML.

## Features
- Fully connected neural network with customizable layer sizes.
- Leaky ReLU activation function for improved gradient flow.
- Training using the MNIST dataset.
- Batch processing for efficient training.
- Learning rate scheduling.
- Simple visualization with SFML.

## Requirements

To build and run the project, ensure you have the following dependencies installed:

- **C++ Compiler:** GCC/Clang/MSVC with C++17 support.
- **SFML:** Version 2.5 or higher (for visualization purposes).
- **MNIST Dataset:** Ensure the dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) are available in the project directory.

## Dataset Preparation

Download the MNIST dataset from [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place the files in the project directory.
