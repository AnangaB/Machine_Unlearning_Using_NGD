from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
import torch.nn as nn

from models.olp import OLP

# return test and train dataset
def get_test_and_train_data(delete_request_size):
    # Data preparation
    mnist_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load trainset
    trainset = datasets.MNIST(root='./data', download=True, train=True, transform=mnist_image_transform)
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True)

    # Dataset sizes
    n = len(trainset)  # Original dataset size

    # Setup Delete Request Loader (simulate deleted data)
    delete_indices = torch.randperm(n)[:delete_request_size]  # Random indices for delete set
    delete_subset = Subset(trainset, delete_indices)
    delete_loader = DataLoader(delete_subset, batch_size=128, shuffle=True)

    # Setup Remaining Data Loader (trainset without delete items)
    all_indices = set(range(n))  # All indices in the original trainset
    remaining_indices = list(all_indices - set(delete_indices.tolist()))  # Complement of delete_indices
    remaining_subset = Subset(trainset, remaining_indices)
    remaining_loader = DataLoader(remaining_subset, batch_size=256, shuffle=True)

    # Load test set
    testset = datasets.MNIST(root='./data', download=True, train=False, transform=mnist_image_transform)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    return train_loader, delete_loader, remaining_loader, test_loader

#train OLP

def get_trained_OLP_model(train_loader, lamd=1e-8):
    """
    Train an OLP model with L2 regularization added explicitly to the loss function.
    
    Args:
        train_loader: DataLoader for the training dataset.
        lamd (float): L2 regularization strength.

    Returns:
        model: Trained OLP model.
    """
    # Model initialization
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_neurons = 200
    output_size = 10  # 10 classes for MNIST digits (0-9)
    model = OLP(input_size, hidden_neurons, output_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.99), eps=10e-8)

    # Modified Training Loop
    epochs = 50 # Number of epochs
    perfect_accuracy = 1.0  # Target 100% accuracy
    near_zero_loss = 1e-6   # Define a threshold for "close to zero" loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Flatten images to (batch_size, input_size)
            images = images.view(-1, input_size)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Add L2 regularization explicitly
            l2_reg = sum(torch.norm(param) ** 2 for param in model.parameters())
            loss = loss + lamd * l2_reg

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics tracking
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Compute average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

        # Check stopping condition
        if accuracy >= perfect_accuracy and avg_loss <= near_zero_loss:
            print("Perfect training accuracy and near-zero loss achieved.")
            break

    return model

# Evaluation function
def evaluate_model(model, test_loader, input_size):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total
