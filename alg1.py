
import numpy as np
import torch

# Noise addition for privacy


def compute_sigma(m,M, C, Q, K, mu, n, epsilon, delta):
    """
    Computes the standard deviation (sigma) for Gaussian noise 
    based on constants and differential privacy parameters.
    
    Parameters:
        m (int): Number of points to 'unlearn'.
        C (float): Lipschitz constant of the gradient.
        Q (float): Smoothness constant of the loss.
        K (float): Bound on the gradient of the features.
        mu (float): Strong convexity parameter.
        n (int): Total number of training data points.
        epsilon (float): Privacy budget.
        delta (float): Failure probability.
    
    Returns:
        sigma (float): Standard deviation for Gaussian noise.
    """
    gamma = (M * m**2 * C**2) / (mu**3 * n**2) + (2 * m**2 * C * Q * K) / (mu**2 * n**2)
    sigma = ((gamma / epsilon) * np.sqrt(2 * np.log(1.25 / delta)))**2
    return sigma

def add_noise(shape, sigma):
    """
    Adds Gaussian noise with the given standard deviation.
    
    Parameters:
        shape (tuple): Shape of the noise tensor.
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        Tensor: Gaussian noise tensor.
    """
    return torch.randn(shape) * sigma

# Partial Algorithm 1 from the paper: Implementation Faster Machine Unlearning via Natural Gradient Descent by Omri Lev and Ashia Wilson
def unlearn(model, delete_loader, original_loader, ngd_optimizer, criterion, n, m, input_size, weight_decay=1e-4, noise_std=1e-3):
    """
    Unlearn using Natural Gradient Descent (NGD) with L2 regularization and noise addition.

    Args:
        model: Neural network model.
        delete_loader: DataLoader for delete request set (U).
        original_loader: DataLoader for original training set (S).
        ngd_optimizer: NGD optimizer for parameter updates.
        criterion: Loss function.
        n: Total number of samples in the original training set (S).
        m: Number of samples in the delete request set (U).
        input_size: Input size of the flattened image.
        weight_decay: L2 regularization coefficient.
        noise_std: Standard deviation for Gaussian noise.
    """
    model.train()
    
    # Step 1: Compute the loss gradient adjustment on delete set
    
    model.zero_grad()  # Clear gradients before computing the gradients for delete set

    for images, labels in delete_loader:
        images = images.view(-1, input_size)  # Flatten input
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Compute gradients for delete set
    
    # Adjust gradients: scale the loss gradients for the delete set
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Apply the scaling factor and add L2 regularization (weight decay)
            #param.grad = (n / (n - m)) * param.grad + weight_decay * param.data  # Regularized gradient
            param.grad = (n / (n - m)) * param.grad 
            
    # Step 2: NGD parameter update (with Fisher/Hessian approximation)
    for images, labels in original_loader:
        images = images.view(-1, input_size)  # Flatten input
        outputs = model(images)
        loss = criterion(outputs, labels)  # Loss for the original set
        
        # NGD optimizer step
        ngd_optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients on the original set
        
        # Add L2 regularization (Hessian of the regularizer)
        for param in model.parameters():
            if param.grad is not None:
                # L2 regularization (Hessian term) - corresponds to λ * param
                param.grad += weight_decay * param.data  # Regularizer's Hessian (λI) contribution
    
    # Perform NGD update step
    ngd_optimizer.step()  # Perform the NGD update step
    # Step 3: Add Gaussian noise for privacy
    with torch.no_grad():
        for param in model.parameters():
            Q = 1          # Smoothness constant
            K = .1           # Gradient bound
            M = 2 *weight_decay + K
            C = 2 *weight_decay + K  #  Lipschitz constant of the gradient of the loss + regularizer
            print(M,C)
            mu = 0.1        # Strong convexity
            epsilon = .5   # Privacy budget
            delta = 1e-5    # Failure probability

            # Compute sigma
            sigma = compute_sigma(m,M, C, Q, K, mu, n, epsilon, delta)
            sigma = .1
            print(f"Computed Sigma: {sigma}")

            noise = add_noise(param.shape, sigma)  # Generate Gaussian noise
            
            param.add_(noise)  # Add noise to the parameters
    
    print("Unlearning completed with NGD, Hessian regularization (L2), and Gaussian noise addition.")
