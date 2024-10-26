from tqdm import tqdm  # For displaying a progress bar during training
import timeit  # For measuring execution time
from main import FNN, tune_hyper_parameter  # Import FNN model and hyperparameter tuning function

class Params:
    """
    Class to hold hyperparameters and configuration for training, validation, and testing.
    """

    # Nested class for batch sizes for train, validation, and test sets
    class BatchSize:
        train = 128
        val = 128
        test = 1000

    def __init__(self):
        """
        Initialize default parameters for training.
        """
        self.mode = 'fnn'  # Mode to execute, either 'fnn' for training or 'tune' for hyperparameter tuning
        # self.model = 'tune'
        self.target_metric = 'accuracy'  # Metric for tuning ('accuracy' or 'loss')
        # self.target_metric = 'loss'

        self.device = 'gpu'  # Device choice, 'gpu' or 'cpu'
        self.loss_type = "ce"  # Loss type, e.g., Cross-Entropy (ce)
        self.batch_size = Params.BatchSize()  # Batch size configuration
        self.n_epochs = 10  # Number of epochs for training
        self.learning_rate = 1e-1  # Learning rate for the optimizer
        self.momentum = 0.5  # Momentum for SGD optimizer


def get_dataloaders(batch_size):
    """
    Create data loaders for the CIFAR-10 dataset.

    Parameters:
    - batch_size (Params.BatchSize): Object holding batch sizes for train, validation, and test sets.

    Returns:
    - train_loader, val_loader, test_loader: Data loaders for training, validation, and test sets.
    """
    import torch
    from torch.utils.data import random_split
    import torchvision

    # Download and transform CIFAR-10 training dataset
    CIFAR_training = torchvision.datasets.CIFAR10(
        root='.', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    # Download and transform CIFAR-10 test dataset
    CIFAR_test_set = torchvision.datasets.CIFAR10(
        root='.', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    # Split training data into training and validation sets
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    # Create data loaders for each dataset split
    train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=batch_size.train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=batch_size.val, shuffle=False)
    test_loader = torch.utils.data.DataLoader(CIFAR_test_set, batch_size=batch_size.test, shuffle=False)

    return train_loader, val_loader, test_loader


def train(net, optimizer, train_loader, device):
    """
    Train the network for one epoch.

    Parameters:
    - net (torch.nn.Module): The model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    - device (torch.device): Device to run the model on (CPU or GPU).
    """
    net.train()  # Set the model to training mode
    pbar = tqdm(train_loader, ncols=100, position=0, leave=True)  # Progress bar
    avg_loss = 0  # Track average loss

    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()  # Reset gradients
        data, target = data.to(device), target.to(device)  # Move data to device
        output = net(data)  # Forward pass
        loss = net.get_loss(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        loss_sc = loss.item()  # Get scalar loss value
        avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)  # Update average loss

        # Update progress bar description
        pbar.set_description(f'train loss: {loss_sc:.6f} avg loss: {avg_loss:.6f}')


def validation(net, validation_loader, device):
    """
    Validate the network on the validation dataset.

    Parameters:
    - net (torch.nn.Module): The model to be validated.
    - validation_loader (torch.utils.data.DataLoader): DataLoader for validation data.
    - device (torch.device): Device to run the model on (CPU or GPU).
    """
    net.eval()  # Set model to evaluation mode
    validation_loss = 0
    correct = 0  # Counter for correct predictions

    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)  # Move data to device
        output = net(data)  # Forward pass
        loss = net.get_loss(output, target)  # Compute loss
        validation_loss += loss.item()  # Accumulate validation loss
        pred = output.data.max(1, keepdim=True)[1]  # Get predicted labels
        correct += pred.eq(target.data.view_as(pred)).sum().item()  # Count correct predictions

    # Compute average loss and accuracy
    validation_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)

    print(f'\nValidation set: Avg. loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n')


def test(net, test_loader, device):
    """
    Test the network on the test dataset.

    Parameters:
    - net (torch.nn.Module): The trained model to be tested.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    - device (torch.device): Device to run the model on (CPU or GPU).
    """
    net.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0  # Counter for correct predictions

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)  # Move data to device
        output = net(data)  # Forward pass
        loss = net.get_loss(output, target)  # Compute loss
        test_loss += loss.item()  # Accumulate test loss
        pred = output.data.max(1, keepdim=True)[1]  # Get predicted labels
        correct += pred.eq(target.data.view_as(pred)).sum().item()  # Count correct predictions

    # Compute average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')


def main():
    """
    Main function to run either training and testing or hyperparameter tuning.
    """
    params = Params()  # Initialize parameters

    # Try importing paramparse for argument parsing
    try:
        import paramparse
    except ImportError:
        print("paramparse is unavailable so command-line arguments will not work")
    else:
        paramparse.process(params)

    # Import necessary PyTorch modules
    import torch
    import torch.optim as optim

    # Set random seed for reproducibility
    random_seed = 1
    torch.manual_seed(random_seed)

    # Determine device (CPU or GPU) for running the model
    device = torch.device("cuda" if params.device != 'cpu' and torch.cuda.is_available() else "cpu")
    print(f'Running on {"GPU: " + torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')
    print(params.mode)

    # Execute function based on the mode parameter
    if params.mode == 'fnn':
        # Initialize data loaders, model, and optimizer
        train_loader, val_loader, test_loader = get_dataloaders(params.batch_size)
        net = FNN(params.loss_type, 10).to(device)  # Initialize model and move to device
        optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum)

        # Measure total runtime of training and testing
        start = timeit.default_timer()

        # Initial validation before training
        with torch.no_grad():
            validation(net, val_loader, device)

        # Training loop over epochs
        for epoch in range(params.n_epochs):
            print(f'\nEpoch {epoch + 1} / {params.n_epochs}\n')
            train_start = timeit.default_timer()

            # Train the model
            train(net, optimizer, train_loader, device)

            train_stop = timeit.default_timer()
            print(f'\nTrain runtime: {train_stop - train_start:.2f} secs')

            # Validate the model after each epoch
            with torch.no_grad():
                validation(net, val_loader, device)

        # Test the model after training
        with torch.no_grad():
            test(net, test_loader, device)

        # Calculate total runtime for the mode
        print(f'Total runtime: {timeit.default_timer() - start:.2f} secs')

    elif params.mode == 'tune':
        # Run hyperparameter tuning
        start = timeit.default_timer()
        print(params.target_metric, device)
        best_params, best_metric = tune_hyper_parameter(params.target_metric, device)
        run_time = timeit.default_timer() - start

        # Display results of hyperparameter tuning
        print(f"\nBest {params.target_metric}: {best_metric}")
        print(f"Best params:\n{best_params}")
        print(f"Runtime of tune_hyper_parameter: {run_time}")
    else:
        raise AssertionError(f'Invalid mode: {params.mode}')


# Run the main function when script is executed directly
if __name__ == "__main__":
    main()
