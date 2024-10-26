# Logistic Regression and Feedforward Neural Network on MNIST and CIFAR-10

This project implements and trains a **Logistic Regression** model and a **Feedforward Neural Network (FNN)** model to classify images from the **MNIST** and **CIFAR-10** datasets, respectively. The project includes functions for training, validating, and hyperparameter tuning both models. Additionally, hyperparameter tuning is automated through a grid search approach.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Code Structure](#code-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Class and Function Descriptions](#class-and-function-descriptions)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Expected Output](#expected-output)

## Project Overview
The goal of this project is to:
- **Train a Logistic Regression model** on the MNIST dataset to classify handwritten digits (0-9).
- **Train a Feedforward Neural Network (FNN) model** on the CIFAR-10 dataset to classify objects into 10 classes (e.g., airplanes, cars, birds).
- **Perform hyperparameter tuning** on both models using a grid search approach.

The code is organized to handle all parts of data loading, model training, validation, and hyperparameter tuning with minimal input, and is structured for easy modification and testing.

## Code Structure

The project is organized as follows:
- **`LogisticRegressionModel` class**: Implements a simple Logistic Regression model using a single linear layer.
- **`FNN` class**: Implements a Feedforward Neural Network with multiple fully connected layers for more complex classification tasks.
- **`Params` class**: Configures and holds training parameters and hyperparameters like learning rate, batch size, number of epochs, and device settings.
- **Training and Validation Functions**: Includes functions for training, validating, and evaluating models on MNIST and CIFAR-10 datasets.
- **Hyperparameter Tuning Function**: Performs grid search on learning rate and batch size for both models, selects the best parameters, and evaluates model accuracy.

## Requirements

To run this project, you need to have the following packages installed:
- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **TQDM** (for progress bars)

## Setup

1. **Clone the repository** and navigate to your project directory.

2. **Prepare data**:
   - The code automatically downloads the **MNIST** and **CIFAR-10** datasets when run.

## Usage

The code provides two main models for classification tasks. Here’s how to train, validate, and perform hyperparameter tuning for both models.

### Running Logistic Regression on MNIST
To train and evaluate the Logistic Regression model:
```python
from your_module import logistic_regression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = logistic_regression(device=device, learning_rate=0.01, batch_size=64)
```

### Running FNN on CIFAR-10
To train and evaluate the Feedforward Neural Network model:
```python
from your_module import train_fnn, Params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = Params()
results = train_fnn(device=device, learning_rate=params.learning_rate, batch_size=params.batch_size, params=params)
```

### Hyperparameter Tuning for Both Models
To perform hyperparameter tuning on both models:
```python
from your_module import tune_hyper_parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_params, best_metric = tune_hyper_parameter(target_metric='accuracy', device=device)
```

## Class and Function Descriptions

### 1. `LogisticRegressionModel`
- **Purpose**: Implements a single-layer logistic regression model for classifying the MNIST dataset.
- **Parameters**:
  - `input_size`: Total input features (784 for 28x28 MNIST images).
  - `num_classes`: Number of output classes (10 for MNIST digits 0-9).
- **Functions**:
  - `forward`: Flattens the input tensor and applies the linear layer.

### 2. `FNN`
- **Purpose**: Implements a multi-layer feedforward neural network for CIFAR-10 image classification.
- **Parameters**:
  - `loss_type`: Specifies the type of loss function (e.g., 'ce' for cross-entropy).
  - `num_classes`: Number of output classes (10 for CIFAR-10 categories).
- **Functions**:
  - `forward`: Implements a forward pass with three layers, using Tanh and ReLU activations.
  - `get_loss`: Computes the cross-entropy loss based on model output and target labels.

### 3. `Params`
- **Purpose**: Stores model and training parameters for easy access and modification.
- **Attributes**:
  - `mode`: Specifies whether to run the 'fnn' model or 'logistic' model.
  - `target_metric`: The metric used for hyperparameter tuning (default is 'accuracy').
  - `device`: Device on which to run the model ('cpu' or 'gpu').
  - `loss_type`: Type of loss function to use.
  - `batch_size`: Batch size for training and validation.
  - `n_epochs`: Number of epochs for training.
  - `learning_rate`: Learning rate for the optimizer.
  - `momentum`: Momentum value for the optimizer.

### 4. Key Functions

#### `logistic_regression(device, learning_rate, batch_size)`
Trains and validates a logistic regression model on MNIST.
- **Parameters**:
  - `device`: Device to use ('cpu' or 'gpu').
  - `learning_rate`: Learning rate for SGD optimizer.
  - `batch_size`: Batch size for training and validation.
- **Returns**: A dictionary containing the trained model.

#### `train_fnn(device, learning_rate, batch_size, params)`
Trains and validates the FNN model on CIFAR-10.
- **Parameters**:
  - `device`: Device to use ('cpu' or 'gpu').
  - `learning_rate`: Learning rate for SGD optimizer.
  - `batch_size`: Batch size configuration from `Params`.
  - `params`: Additional parameters (e.g., epochs).
- **Returns**: A dictionary containing the trained model and validation loader.

#### `validate_reg_model(model, batch_size, device)`
Validates the logistic regression model on the MNIST validation dataset.
- **Parameters**:
  - `model`: The trained logistic regression model.
  - `batch_size`: Batch size for validation data.
  - `device`: Device to run the model on.
- **Returns**: Validation accuracy as a decimal value.

#### `validation_fnn(net, validation_loader, device)`
Validates the FNN model on the CIFAR-10 validation set.
- **Parameters**:
  - `net`: The trained FNN model.
  - `validation_loader`: DataLoader for the validation set.
  - `device`: Device to run the model on.
- **Returns**: Validation accuracy as a percentage.

## Hyperparameter Tuning

The `tune_hyper_parameter` function performs grid search over learning rates and batch sizes to find the best configuration for both models.

#### `tune_hyper_parameter(target_metric, device)`
Performs hyperparameter tuning for the logistic regression and FNN models.
- **Parameters**:
  - `target_metric`: The metric to optimize (default is 'accuracy').
  - `device`: Device to run the models on.
- **Returns**:
  - `best_params`: Best parameters found for each model.
  - `best_metric`: The highest metric score achieved for each model.

## Expected Output
The output will display accuracy, score, and runtime for the logistic regression model:
```console
Results on logistic mode:
        accuracy: 0.9206
        score: 90.59999999999994
        run_time: 149.3864571
```
### Explanation of Output
- **Accuracy**: Represents the model’s prediction accuracy on the test dataset.
- **Score**: A standardized metric based on accuracy thresholds, showing how well the model performs on a scale from 0-100 scaling linearly with accuracy.
- **Runtime**: Total time taken to train and test the model, providing insight into the efficiency of the training process.

The output will display accuracy, average loss, and runtime for the FNN model:
```console
Running on CPU
fnn
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to .\cifar-10-python.tar.gz
100%|████████████████████████████████████████████| 170498071/170498071 [01:14<00:00, 2291957.67it/s]
Extracting .\cifar-10-python.tar.gz to .
Files already downloaded and verified

Validation set: Avg. loss: 0.0182, Accuracy: 1083/10000 (10.83%)


Epoch 1 / 10

train loss: 2.216860 avg loss: 2.222976: 100%|████████████████████| 313/313 [00:09<00:00, 32.03it/s] 

Train runtime: 9.77 secs

Validation set: Avg. loss: 0.0171, Accuracy: 2902/10000 (29.02%)


Epoch 2 / 10

train loss: 2.103729 avg loss: 2.143075: 100%|████████████████████| 313/313 [00:09<00:00, 32.32it/s] 

Train runtime: 9.69 secs

Validation set: Avg. loss: 0.0169, Accuracy: 3218/10000 (32.18%)


Epoch 3 / 10

train loss: 2.024887 avg loss: 2.103565: 100%|████████████████████| 313/313 [00:09<00:00, 31.72it/s] 

Train runtime: 9.87 secs

Validation set: Avg. loss: 0.0166, Accuracy: 3647/10000 (36.47%)


Epoch 4 / 10

train loss: 2.114366 avg loss: 2.071769: 100%|████████████████████| 313/313 [00:09<00:00, 32.72it/s] 

Train runtime: 9.57 secs

Validation set: Avg. loss: 0.0164, Accuracy: 3805/10000 (38.05%)


Epoch 5 / 10

train loss: 1.990851 avg loss: 2.054358: 100%|████████████████████| 313/313 [00:09<00:00, 31.47it/s] 

Train runtime: 9.95 secs

Validation set: Avg. loss: 0.0163, Accuracy: 3858/10000 (38.58%)


Epoch 6 / 10

train loss: 1.951337 avg loss: 2.040903: 100%|████████████████████| 313/313 [00:09<00:00, 32.00it/s] 

Train runtime: 9.78 secs

Validation set: Avg. loss: 0.0163, Accuracy: 3959/10000 (39.59%)


Epoch 7 / 10

train loss: 1.977517 avg loss: 2.030341: 100%|████████████████████| 313/313 [00:09<00:00, 31.96it/s] 

Train runtime: 9.79 secs

Validation set: Avg. loss: 0.0162, Accuracy: 3990/10000 (39.90%)


Epoch 8 / 10

train loss: 2.016930 avg loss: 2.019848: 100%|████████████████████| 313/313 [00:09<00:00, 33.08it/s] 

Train runtime: 9.46 secs

Validation set: Avg. loss: 0.0162, Accuracy: 4094/10000 (40.94%)


Epoch 9 / 10

train loss: 1.949650 avg loss: 2.010774: 100%|████████████████████| 313/313 [00:09<00:00, 32.75it/s] 

Train runtime: 9.56 secs

Validation set: Avg. loss: 0.0162, Accuracy: 4129/10000 (41.29%)


Epoch 10 / 10

train loss: 1.937896 avg loss: 2.001265: 100%|████████████████████| 313/313 [00:09<00:00, 31.91it/s] 

Train runtime: 9.81 secs

Validation set: Avg. loss: 0.0161, Accuracy: 4180/10000 (41.80%)


Test set: Avg. loss: 0.0020, Accuracy: 4239/10000 (42.39%)

Total runtime: 123.44 secs
```

## Notes on Implementation
- **Normalization**: Normalization is applied to all datasets to ensure consistent input distribution.
- **Loss Function**: Cross-entropy loss is used for both logistic regression and FNN to handle multi-class classification effectively.
- **Optimizer**: SGD is chosen for simplicity and effectiveness, with options for tuning learning rates and batch sizes.

## Dependencies
- **PyTorch**: For model definition, training, and evaluation.
- **Torchvision**: For loading the MNIST dataset.
- **TQDM**: For progress bar display during training.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.