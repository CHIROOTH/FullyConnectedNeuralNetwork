from tqdm import tqdm  # For displaying progress bars
import timeit  # For measuring runtime
import itertools  # For creating hyperparameter grid search combinations
import torch  # PyTorch library
import torch.nn as nn  # Neural network module in PyTorch
import torch.nn.functional as F  # Common activation functions
import torch.optim as optim  # Optimization algorithms like SGD
import torchvision  # For popular datasets
from torchvision import datasets, transforms  # For loading and transforming data
from torch.utils.data import DataLoader, random_split  # For creating data loaders and splits


# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Initializes a simple logistic regression model with a single linear layer.

        Parameters:
        - input_size (int): Number of input features (e.g., 28*28 for MNIST).
        - num_classes (int): Number of output classes (e.g., 10 for MNIST digits).
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # Linear layer for logistic regression

    def forward(self, x):
        """
        Forward pass: Flattens input to 2D (batch_size, input_size) and applies linear layer.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (batch_size, 28, 28) for MNIST images.

        Returns:
        - torch.Tensor: The output predictions with shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten input from [batch_size, 28, 28] to [batch_size, 784]
        return self.linear(x)


def logistic_regression(device, learning_rate=0.01, batch_size=64):
    """
    Train and validate a logistic regression model on the MNIST dataset.

    Parameters:
    - device (torch.device): Device to run the model on (CPU or GPU).
    - learning_rate (float): Learning rate for the optimizer.
    - batch_size (int): Batch size for data loaders.

    Returns:
    - dict: Contains the trained model.
    """
    input_size = 28 * 28  # MNIST image dimensions (28x28 pixels)
    num_classes = 10  # Number of classes in MNIST (digits 0-9)
    num_epochs = 10  # Number of training epochs

    # Data transformations: Normalize images based on MNIST dataset stats
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Split training dataset into training and validation sets
    train_size = len(train_dataset) - 12000  # Remaining samples for training
    val_size = 12000  # Set aside 12000 samples for validation
    train_dataset, val_dataset = torch.utils.data.Subset(train_dataset, range(train_size)), torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))

    # Data loaders for training, validation, and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = LogisticRegressionModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent optimizer

    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, input_size).to(device)  # Flatten and send data to device
            target = target.to(device)  # Send target to device

            # Forward pass and compute loss
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase after each epoch
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct = 0  # Counter for correct predictions
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for data, target in val_loader:
                data = data.view(-1, input_size).to(device)  # Flatten input
                target = target.to(device)
                outputs = model(data)

                # Accumulate loss and correct predictions
                val_loss += criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True)  # Predicted class with max score
                correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

        # Calculate average loss and accuracy for validation set
        val_loss /= len(val_loader.dataset)  # Average validation loss
        accuracy = 100. * correct / len(val_loader.dataset)  # Validation accuracy percentage

    # Return the trained model
    return dict(model=model)


# Feedforward Neural Network (FNN) Model
class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        """
        Initialize the feedforward neural network (FNN) model.

        Parameters:
        - loss_type (str): Type of loss function to use (e.g., 'ce' for cross-entropy).
        - num_classes (int): Number of output classes.
        """
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes

        # Define layers in the FNN model
        self.fc1 = nn.Linear(32 * 32 * 3, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)  # Second fully connected layer
        self.fc3 = nn.Linear(32, num_classes)  # Output layer with 'num_classes' outputs

        # Initialize loss function
        if loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass of the FNN model.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (batch_size, 32, 32, 3) for CIFAR-10 images.

        Returns:
        - torch.Tensor: Output tensor with class scores.
        """
        x = x.view(x.size(0), -1)  # Flatten input to shape (batch_size, 32*32*3)
        x = torch.tanh(self.fc1(x))  # First layer with tanh activation
        x = F.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)  # Output layer (logits for class scores)
        return F.softmax(x, dim=1)  # Softmax activation for class probabilities

    def get_loss(self, output, target):
        """
        Compute the loss based on model outputs and ground truth targets.

        Parameters:
        - output (torch.Tensor): Model output logits.
        - target (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed loss value.
        """
        return self.loss_fn(output, target)  # Compute cross-entropy loss


# Hyperparameter tuning function
def tune_hyper_parameter(target_metric, device):
    """
    Perform hyperparameter tuning for logistic regression and FNN using grid search.

    Parameters:
    - target_metric (str): Metric for tuning (e.g., 'accuracy').
    - device (torch.device): Device to run models on.

    Returns:
    - tuple: Best parameters and corresponding metric.
    """
    # Define hyperparameter search space for grid search
    learning_rates = [0.01, 0.1]
    batch_sizes = [128, 64]
    params = Params()  # Initialize default parameters

    # Initialize best parameter tracking for logistic regression and FNN
    best_params = [{"logistic_regression": {"learning_rate": None, "batch_size": None}},
                   {"FNN": {"learning_rate": None, "batch_size": None}}]
    best_metric = [{"logistic_regression": {"accuracy": None}},
                   {"FNN": {"accuracy": None}}]

    # Hyperparameter search for Logistic Regression model
    log_start = timeit.default_timer()  # Start timing logistic regression tuning
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        model_results = logistic_regression(device, learning_rate=lr, batch_size=bs)  # Train model
        accuracy = validate_reg_model(model_results['model'], bs, device)  # Calculate validation accuracy

        # Update best parameters if accuracy improves
        if best_metric[0]['logistic_regression']['accuracy'] is None or accuracy > best_metric[0]['logistic_regression']['accuracy']:
            best_metric[0]['logistic_regression']['accuracy'] = accuracy
            best_params[0]['logistic_regression']['learning_rate'] = lr
            best_params[0]['logistic_regression']['batch_size'] = bs

    log_stop = timeit.default_timer()  # End timing logistic regression tuning

    # Hyperparameter search for FNN model
    fnn_start = timeit.default_timer()  # Start timing FNN tuning
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        params.learning_rate, params.batch_size.train, params.batch_size.val = lr, bs, bs  # Set parameters
        fnn_results = train_fnn(device, lr, params.batch_size, params)  # Train FNN model
        accuracy = validation_fnn(fnn_results['net'], fnn_results['val_loader'], device)  # Validate FNN

        # Update best parameters if accuracy improves
        if best_metric[1]['FNN']['accuracy'] is None or accuracy > best_metric[1]['FNN']['accuracy']:
            best_metric[1]['FNN']['accuracy'] = accuracy
            best_params[1]['FNN']['learning_rate'] = lr
            best_params[1]['FNN']['batch_size'] = bs

    fnn_stop = timeit.default_timer()  # End timing FNN tuning
    runtime = (fnn_stop - fnn_start) + (log_stop - log_start)  # Total tuning runtime

    return best_params, best_metric  # Return best parameters and metrics


# Validate function for logistic regression model
def validate_reg_model(model, batch_size, device):
    """
    Validate the logistic regression model on the MNIST validation dataset.

    Parameters:
    - model (nn.Module): Trained logistic regression model.
    - batch_size (int): Batch size for data loading.
    - device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
    - float: Validation accuracy as a decimal.
    """
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset and create validation split
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_size = len(train_dataset) - 12000
    val_size = 12000
    _, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Validation data loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Model evaluation
    model.eval()
    num_correct = 0
    total = 0
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    return num_correct / total  # Validation accuracy


# Train FNN model
def train_fnn(device, learning_rate, batch_size, params):
    """
    Train the FNN model on CIFAR-10 dataset.

    Parameters:
    - device (torch.device): Device to run the model on.
    - learning_rate (float): Learning rate for optimizer.
    - batch_size (Params.BatchSize): Batch sizes for training and validation.
    - params (Params): Additional training parameters (e.g., epochs, momentum).

    Returns:
    - dict: Contains trained FNN model and validation loader.
    """
    # Load CIFAR-10 dataset and create training/validation split
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    CIFAR_training = datasets.CIFAR10(root='.', train=True, download=True, transform=transform)
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    # Data loaders
    train_loader = DataLoader(CIFAR_train_set, batch_size=batch_size.train, shuffle=True)
    val_loader = DataLoader(CIFAR_val_set, batch_size=batch_size.val, shuffle=False)

    # Initialize model, optimizer
    net = FNN(params.loss_type, 10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum)

    # Training loop
    for epoch in range(params.n_epochs):
        net.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = net.get_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation after each epoch
        validation_fnn(net, val_loader, device)

    return {'net': net, 'val_loader': val_loader}  # Return trained model and validation loader


# Validate FNN model
def validation_fnn(net, validation_loader, device):
    """
    Evaluate the FNN model on the validation set.

    Parameters:
    - net (nn.Module): Trained FNN model.
    - validation_loader (DataLoader): Validation data loader.
    - device (torch.device): Device to run the model on.

    Returns:
    - float: Validation accuracy percentage.
    """
    net.eval()  # Set model to evaluation mode
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return 100. * correct / len(validation_loader.dataset)  # Validation accuracy percentage


# Parameters class for FNN model and hyperparameter tuning
class Params:
    class BatchSize:
        train = 128
        val = 128

    def __init__(self):
        """
        Initialize parameters for training, validation, and testing.
        """
        self.mode = 'fnn'  # Mode for running the model (fnn or logistic)
        self.target_metric = 'accuracy'  # Metric for tuning (accuracy or loss)
        self.device = 'gpu'  # Device to run model on (gpu or cpu)
        self.loss_type = "ce"  # Loss type (cross-entropy in this case)
        self.batch_size = Params.BatchSize()  # Batch sizes for train and validation
        self.n_epochs = 10  # Number of epochs for training
        self.learning_rate = 1e-1  # Learning rate for optimizer
        self.momentum = 0.5  # Momentum for SGD optimizer
