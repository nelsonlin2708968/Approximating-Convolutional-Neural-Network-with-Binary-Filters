# Approximating Convolutional Neural Network with Binary Filters

This project focuses on the partial implementation of the article [Towards Accurate Binary Convolutional Neural Network (Xiofan Lin, Cong Zhao, Wei Pan)](https://arxiv.org/abs/1711.11294). The need for faster and more cost efficient testing in CNNs motivated the authors to build a binary neural network achieving prediction accuracy comparable to it's full-precision counterpart. Two major innovations are presented in the paper: (1) Approximation of full-precision weights with the linear combinations of multiple binary weights; (2) Multiple binary activations. In this project, we implement (1) on CIFAR-10.

## Instructions

The main notebook is in `approx_cnn_notebook.ipynb`. It goes through how to load the data, how to train a full-weigh CNN on the data and use this pretrained model to train an approximated CNN using binary filters.
You can load the raw CIFAR-10 data using the `load_data` function from `cifar_utils.py`.
The `approx_cnn.py` file contains all the necessary classes and functions to build and train our networks using tensorflow.
The `cnn_jupyter_tensorboard.py` file contains the necessary functions to retrieve and plot the tensorflow loss graphs of our CNN's. 

The models and logs are saved in the respective folders `model` and `log`

## Authors


* **Nelson Lin**
* **Marianne Sorba** 
