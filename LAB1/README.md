# Laboratory #1

## Overview
This laboratory aims to recreate on a small scale in PyTorch the results obtained in the paper that introduced ResNets
> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.

in order to demonstrate that these residual networks are easier to optimize and can gain accuracy from increased depth, compared to plain networks, addressing the degradation problem of the training error and the problem of vanishing/exploding gradients.

## Data
The datasets used are MNIST (gray scale images of handwritten digits from 0 to 9) and CIFAR10 (60000 RGB images categorized in airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck) available from `torchvision.datasets`, both suitable for 10-class classification task.

## Code description
In [config.py](src/config.py) an `ArgumentParser` is employed to easily configurate in the command line the model optimization and evaluation parameters.

[models.py](src/models.py) implements a collection of neural network models, ranging from classic Multi-Layer Perceptrons to more advanced architectures like Residual Networks and Convolutional Neural Networks, designed for image classification task and to serve as a foundation for further experimentations and observation.

* `MLP_2layers`: a basic MLP with two fully connected (linear) layers;
* `MLP_3layers`: an extension of the basic MLP with an additional hidden layer, featuring GeLU activation functions for enhanced performance;
* `DynamicMLP`: a flexible and adaptable MLP that can be instantiated with a variable number of hidden layers and custom sizes;
* `DynamicMLP_improved`: an enhanced version of the dynamic MLP that incorporates regularization techniques such as `BatchNorm1d` and `Dropout` to improve training stability;
* `ResidualMLP`: a MLP built with residual blocks, leveraging **skip connections** to facilitate the training of deeper networks;
* `DynamicResidualMLP`: a fully dynamic version of the Residual MLP, allowing for the creation of residual networks with a variable number of layers and a custom block structure;
* `myCNN`: a simple CNN architecture, featuring two convolutional layers followed by linear layers;
* `myCNN_improved`: an advanced CNN model that includes regularization techniques like `BatchNorm2d`, and `Dropout` at the linear layers level, along with a deeper convolutional backbone, for improved accuracy and robustness.

In [trainer.py](src/trainer.py) is defined the `trainer()` function to train and validate the models, logging on the `Weights & Biases` platform the running metrics of `train/loss`, `val/loss`, `val/accuracy`, `train/lr` to visualize the evolution of their performances during the epochs.

In [tester.py](src/tester.py) is defined the `tester()` function to test the models and output the testing average loss and accuracy values, quantifying the correctness of the network predictions.

The [main.py](src/main.py) script serves as the primary entry point for training and evaluating various deep learning models on either the **MNIST** or **CIFAR-10** datasets, providing a modular workflow for selecting a model, setting up the data loaders, running the training and testing loops, and saving the results.

1. The script uses the [config.py](src/config.py) file to manage all hyperparameters and settings, including the model, dataset, learning rate, and batch size;
2. dynamically loads the specified model and dataset based on the configuration, supporting both custom-defined architectures and pre-trained models like ResNet18;
3. It manages the complete machine learning workflow:
    * [trainer.py](src/trainer.py) that manages the training loop, including loss calculation, backpropagation, and logging;
    * [tester.py](src/tester.py) evaluates the final model on the test set;
4. tracks and plots key performance metrics such as training loss, validation loss, and validation accuracy over epochs;
5. the best-performing model (based on validation accuracy) is automatically saved in the [models](/models) folder, ensuring that the optimal weights are preserved for testing and possible later use.


## Exercises

### Exercise 1.1 – A baseline MLP
> Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two narrow layers). Implement your own training pipeline. Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch.

| Training Loss | Validation Loss | Validation Accuracy | 
|:-----------------:|:---------------------:|:---------------------:|
| ![ Training Loss ](images/first_mlp_train.png) | ![Validation Loss](images/first_mlp_val.png) | ![Validation Acc](images/first_mlp_val.png) |

### Exercise 1.2 – Adding Residual Connections
> Implement a variant of your parameterized MLP network to support residual connections. Your network should be defined as a composition of residual MLP blocks that have one or more linear layers and add a skip connection from the block input to the output of the final linear layer. Compare the performance (in training/validation loss and test accuracy) of your MLP and ResidualMLP for a range of depths. Verify that deeper networks with residual connections are easier to train than a network of the same depth without residual connections.

| Model                         | Hidden Layers Size               | Test Accuracy |
|-------------------------------|----------------------------------|:-------------:|
| `Dynamic_MLP_2layers`         | [128, 64]                        | 0.8658        |
| `Dynamic_MLP_4layers`         | [128, 64, 64, 32]                | 0.0991        |
| `Dynamic_MLP_8layers`         | [128, 128, 64, 64, 32, 32, 16, 16]| 0.1010       |
| `Dynamic_improved_MLP_2layers`| [128, 64]                        | 0.9204        |
| `Dynamic_improved_MLP_4layers`| [128, 64, 64, 32]                | 0.9151        |
| `Dynamic_improved_MLP_8layers`| [128, 128, 64, 64, 32, 32, 16, 16]| 0.6887       |
| `Residual_MLP_2layers`        | [128, 64]                        | 0.9332        |
| `Residual_MLP_4layers`        | [128, 64, 64, 32]                | **0.9416**    |
| `Residual_MLP_8layers`        | [128, 128, 64, 64, 32, 32, 16, 16]| 0.9313       |



### Exercise 1.3 – Rinse and Repeat (but with a CNN)
> Repeat the verification you did above, but with **Convolutional** Neural Networks and using CIFAR-10. Show that **deeper** CNNs *without* residual connections do not always work better and **even deeper** ones *with* residual connections. 

| Model                         | ?               | Test Accuracy |
|-------------------------------|----------------------------------|:-------------:|
| `myCNN`         | [128, 64]                        | 0.        |
| `myCNN_improved`         | [128, 64, 64, 32]                | 0.        |
| `ResNet18`         | [128, 128, 64, 64, 32, 32, 16, 16]| 0.6823      |
| `ResNet50`| [128, 64]                        | 0.7403        |

### Exercise 2.? –

