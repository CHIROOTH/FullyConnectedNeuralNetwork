import timeit  # Module to measure execution time
from collections import OrderedDict  # Ordered dictionary to store results in an organized way
import torch  # PyTorch library
from torchvision import transforms, datasets  # For dataset handling and transformations
from main import logistic_regression  # Import logistic regression function from main

# Set multiprocessing sharing strategy to 'file_system' for compatibility across processes
torch.multiprocessing.set_sharing_strategy('file_system')


def compute_score(acc, acc_thresh):
    """
    Compute score based on the accuracy threshold.

    Parameters:
    - acc (float): The accuracy obtained.
    - acc_thresh (tuple): The minimum and maximum thresholds for the score.

    Returns:
    - score (float): Computed score in the range [0, 100].
    """
    min_thres, max_thres = acc_thresh
    if acc <= min_thres:
        score = 0.0
    elif acc >= max_thres:
        score = 100.0
    else:
        score = float(acc - min_thres) / (max_thres - min_thres) * 100
    return score


def test(model, device):
    """
    Evaluate the logistic regression model on the MNIST test dataset.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - device (torch.device): The device (CPU or GPU) on which the model runs.

    Returns:
    - acc (float): The calculated accuracy of the model on the test set.
    """
    # Define MNIST test dataset with required transformations
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # DataLoader for efficient data handling
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    model.eval()  # Set model to evaluation mode
    num_correct = 0  # Counter for correct predictions
    total = 0  # Total number of samples

    # Loop over each test sample
    for batch_idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(device), targets.to(device)

        # Disable gradient calculation for faster evaluation
        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)  # Get the predicted label
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()  # Count correct predictions

    acc = float(num_correct) / total  # Calculate accuracy
    return acc


class Args:
    """
    Define command-line arguments as a class structure.
    This allows setting various parameters for model selection, tuning, and device configuration.
    """
    mode = 'logistic'  # Mode to run ('logistic' for logistic regression, 'tune' for hyperparameter tuning)
    target_metric = 'acc'  # Metric for tuning ('acc' for accuracy, 'loss' for validation loss)
    gpu = 1  # Set to 1 for GPU, 0 for CPU


def main():
    """
    Main function to execute logistic regression and evaluate its performance.
    """
    args = Args()  # Initialize argument structure
    try:
        import paramparse
        paramparse.process(args)  # Process arguments if paramparse is available
    except ImportError:
        pass  # Ignore if paramparse isn't installed

    # Define device based on GPU availability and args.gpu setting
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Accuracy thresholds for scoring based on the mode
    acc_thresh = dict(logistic=(0.83, 0.93))

    if args.mode == 'logistic':  # Run logistic regression if mode is 'logistic'
        start = timeit.default_timer()  # Start timer
        results = logistic_regression(device)  # Call logistic regression function
        model = results['model']  # Retrieve the trained model from results

        # Check if model training was successful
        if model is None:
            print('Model training failed.')
            return

        stop = timeit.default_timer()  # Stop timer
        run_time = stop - start  # Calculate runtime

        # Test model and compute accuracy on the test dataset
        accuracy = test(model, device)

        # Compute score based on accuracy thresholds
        score = compute_score(accuracy, acc_thresh[args.mode])

        # Organize results in a dictionary
        result = OrderedDict(
            accuracy=accuracy,
            score=score,
            run_time=run_time
        )

        # Print results in a structured format
        print(f"Results on {args.mode} mode:")
        for key in result:
            print(f"\t{key}: {result[key]}")


# Run main function if this script is executed directly
if __name__ == "__main__":
    main()
